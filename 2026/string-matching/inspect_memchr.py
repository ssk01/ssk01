#!/usr/bin/env python3
"""Follow the benchmark's memchr call into the loaded macOS implementation."""
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('This inspection targets the macOS arm64 benchmark.')
    subprocess.run(['make', 'all', 'ARCH=arm64'], cwd=ROOT, check=True)
    if not (ROOT/'build/comments.bin').exists():
        from run import sample
        sample()
    commands = [
        'breakpoint set --name libc_c',
        'run blob build/comments.bin build/lengths.bin youtube',
        'script print("BEGIN_MEMCHR_INSPECTION")',
        'disassemble --frame',
        # Locate the actual call instruction instead of assuming a compiler offset.
        'script t=lldb.debugger.GetSelectedTarget(); f=t.GetProcess().GetSelectedThread().GetSelectedFrame(); ins=f.GetSymbol().GetInstructions(t); calls=[i for i in ins if i.GetMnemonic(t)=="bl" and "memchr" in (i.GetComment(t) or "")]; assert len(calls)==1; bp=t.BreakpointCreateByAddress(calls[0].GetAddress().GetLoadAddress(t)); print("Call breakpoint:",bp)',
        'continue',
        'thread step-inst',  # BL into executable's memchr stub.
        'thread step-inst',  # ADRP
        'thread step-inst',  # LDR resolved function pointer
        'thread step-inst',  # BR into the system library
        'image lookup --address $pc',
        'register read x1 x2',
        'disassemble --frame',
        'thread step-inst --count 8',  # Execute through the first vector subtraction.
        'disassemble --start-address $pc --count 4',
        'script print("END_MEMCHR_INSPECTION")',
        'process kill',
    ]
    args=['lldb','--batch','./build/demo']
    for command in commands:
        args.extend(['-o',command])
    result=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=True)
    # Keep the inspected call chain; exclude startup and debugger-perturbed timings.
    lines=result.stdout.splitlines()
    start=lines.index('BEGIN_MEMCHR_INSPECTION')+1
    end=lines.index('END_MEMCHR_INSPECTION')
    evidence='\n'.join(line.rstrip() for line in lines[start:end])
    evidence=evidence.replace(str(ROOT),'.').replace(str(Path.home()),'~')
    assert 'libsystem_platform.dylib`_platform_memchr' in evidence
    for instruction in ('dup.16b','ldr    q0','sub.16b','uminv.16b'):
        assert instruction in evidence, instruction
    assert 'Traceback' not in result.stdout+result.stderr
    version=subprocess.run(['sw_vers'],capture_output=True,text=True,check=True).stdout
    header='Runtime memchr inspection\nArchitecture: arm64\n'+version+'\nCommand: python3 inspect_memchr.py\nNo timings from this debugger run are used in the benchmark results.\n\n'
    (ROOT/'results/memchr-inspection.txt').write_text(header+evidence+'\n')
    print('Confirmed: benchmark call -> libsystem_platform.dylib::_platform_memchr; NEON 16-byte search. Evidence: results/memchr-inspection.txt')

if __name__=='__main__':
    main()
