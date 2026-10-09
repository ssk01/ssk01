import Foundation
import Metal

// Standalone feasibility experiment; not integrated into the TF demo.
let source = """
#include <metal_stdlib>
using namespace metal;
constant uint features [[function_constant(0)]];
kernel void specialized(const device float* a [[buffer(0)]],
                        const device float* b [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        uint row [[thread_position_in_grid]]) {
    float sum = 0;
    for (uint f = 0; f < features; ++f) sum += a[row * features + f] * b[f];
    y[row] = sum;
}
kernel void generic(const device float* a [[buffer(0)]],
                    const device float* b [[buffer(1)]],
                    device float* y [[buffer(2)]],
                    constant uint& features [[buffer(3)]],
                    uint row [[thread_position_in_grid]]) {
    float sum = 0;
    for (uint f = 0; f < features; ++f) sum += a[row * features + f] * b[f];
    y[row] = sum;
}
"""

func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
func ms(_ ns: UInt64) -> Double { Double(ns) / 1_000_000 }
func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("Metal unavailable")
}
let options = MTLCompileOptions()
options.mathMode = .safe
let library = try device.makeLibrary(source: source, options: options)
func create(_ features: Int) throws -> MTLComputePipelineState {
    var f = UInt32(features)
    let constants = MTLFunctionConstantValues()
    constants.setConstantValue(&f, type: .uint, index: 0)
    let function = try library.makeFunction(name: "specialized", constantValues: constants)
    return try device.makeComputePipelineState(function: function)
}
let rows = 64
let shapes = [32, 128, 512]
let requests = 120
let rounds = 7
struct Buffers { let a: MTLBuffer; let b: MTLBuffer; let y: MTLBuffer }
var buffers: [Int: Buffers] = [:]
for f in shapes {
    let a = device.makeBuffer(length: rows * f * 4, options: .storageModeShared)!
    let b = device.makeBuffer(length: f * 4, options: .storageModeShared)!
    let y = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
    let ap = a.contents().bindMemory(to: Float.self, capacity: rows * f)
    let bp = b.contents().bindMemory(to: Float.self, capacity: f)
    for i in 0..<(rows * f) { ap[i] = Float((i % 17) - 8) / 16 }
    for i in 0..<f { bp[i] = Float((i % 11) - 5) / 8 }
    buffers[f] = Buffers(a: a, b: b, y: y)
}
let generic = try device.makeComputePipelineState(function: library.makeFunction(name: "generic")!)
print("device=\(device.name), rows=\(rows), features=\(shapes), requests=\(requests), rounds=\(rounds)")
for f in shapes {
    let start = now()
    _ = try create(f)
    print(String(format: "first specialization F=%d: %.3f ms (driver cache may already exist)", f, ms(now() - start)))
}

struct Result {
    var prep: Double = 0; var execute: Double = 0; var total: Double = 0
    var creates = 0; var hits = 0; var evictions = 0
    var maxError: Float = 0; var checksum: Double = 0
}
func run(_ mode: String, _ capacity: Int) throws -> Result {
    var cache: [Int: MTLComputePipelineState] = [:]
    var lru: [Int] = []
    var result = Result()
    for request in 0..<requests {
        let f = shapes[request % shapes.count]
        let buf = buffers[f]!
        // Change a weight each request: a cached output would be incorrect.
        let bp = buf.b.contents().bindMemory(to: Float.self, capacity: f)
        bp[0] = Float(request % 19) / 16
        let start = now()
        let pipeline: MTLComputePipelineState
        if mode == "generic-global" {
            pipeline = generic
        } else if let hit = cache[f] {
            pipeline = hit
            lru.removeAll { $0 == f }; lru.append(f)
            result.hits += 1
        } else {
            pipeline = try create(f)
            result.creates += 1
            if capacity > 0 {
                if cache.count == capacity {
                    cache.removeValue(forKey: lru.removeFirst())
                    result.evictions += 1
                }
                cache[f] = pipeline; lru.append(f)
            }
        }
        let prepared = now()
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buf.a, offset: 0, index: 0)
        encoder.setBuffer(buf.b, offset: 0, index: 1)
        encoder.setBuffer(buf.y, offset: 0, index: 2)
        if mode == "generic-global" {
            var featureCount = UInt32(f)
            encoder.setBytes(&featureCount, length: 4, index: 3)
        }
        encoder.dispatchThreads(MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(rows, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        if command.status != .completed { fatalError("GPU execution failed: \(String(describing: command.error))") }
        let end = now()
        result.prep += ms(prepared - start)
        result.execute += ms(end - prepared)
        result.total += ms(end - start)
        // Correctness and checksum stay outside timed region.
        let ap = buf.a.contents().bindMemory(to: Float.self, capacity: rows * f)
        let yp = buf.y.contents().bindMemory(to: Float.self, capacity: rows)
        for row in 0..<rows {
            var reference: Float = 0
            for feature in 0..<f { reference += ap[row * f + feature] * bp[feature] }
            let error = abs(reference - yp[row])
            if !yp[row].isFinite || error > 1e-4 { fatalError("Wrong output: F=\(f), row=\(row), error=\(error)") }
            result.maxError = max(result.maxError, error)
            result.checksum += Double(yp[row])
        }
    }
    return result
}
let modes = [("generic-global", 0), ("specialized-off", 0), ("specialized-cache3", 3), ("specialized-cache2", 2)]
// Warm GPU command submission and both execution styles before measurement.
_ = try run("generic-global", 0)
_ = try run("specialized-cache3", 3)
var results: [String: [Result]] = [:]
for round in 0..<rounds {
    let order = round % 2 == 0 ? modes : Array(modes.reversed())
    for (mode, capacity) in order {
        results[mode, default: []].append(try run(mode, capacity))
    }
}
for (mode, _) in modes {
    let rs = results[mode]!
    let r = rs[0]
    print(String(format: "%@: prep=%.3f ms, execute_wait=%.3f ms, total=%.3f ms, creates=%d, hits=%d, evictions=%d, max_error=%.8f, checksum=%.6f",
        mode, median(rs.map { $0.prep }), median(rs.map { $0.execute }), median(rs.map { $0.total }),
        r.creates, r.hits, r.evictions, rs.map { $0.maxError }.max()!, r.checksum))
}
