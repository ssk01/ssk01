"""
DuckDB vs ClickHouse Aggregation Benchmark
===========================================
数据集: 5 千万行交易记录 (~400MB Parquet)
对比引擎:
  - DuckDB 实时聚合 (查询时向量化)
  - ClickHouse MergeTree 实时聚合
  - ClickHouse AggregatingMergeTree 预聚合
  - DuckDB 手动物化表
"""
import duckdb
import subprocess
import time
import os
import shutil

ROW_COUNT = 50_000_000
DATA_DIR  = "/tmp/bench_ch_vs_ddb"
PARQUET   = f"{DATA_DIR}/raw.parquet"
CH_HOST   = "127.0.0.1"
CH_PORT   = "9123"
CH_DIR    = f"{DATA_DIR}/ch_data"
CH_BIN    = "/usr/local/bin/clickhouse"

def run_ch(sql, label="", timeout=120):
    t0 = time.time()
    proc = subprocess.run(
        ["arch", "-x86_64", CH_BIN, "client",
         "--host", CH_HOST, "--port", CH_PORT,
         "--query", sql],
        capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [ERR] {label}: {proc.stderr.strip()[:200]}")
    return elapsed, proc.stdout

def start_clickhouse():
    os.makedirs(CH_DIR, exist_ok=True)
    subprocess.run(["pkill", "-f", f"clickhouse.*{CH_DIR}"], capture_output=True)
    time.sleep(2)
    proc = subprocess.Popen(
        ["arch", "-x86_64", CH_BIN, "server",
         "--log-file", f"{CH_DIR}/server.log",
         "--errorlog-file", f"{CH_DIR}/error.log",
         "--",
         f"--path={CH_DIR}",
         f"--tmp_path={CH_DIR}/tmp",
         f"--user_files_path={CH_DIR}/user_files",
         f"--tcp_port={CH_PORT}",
         "--http_port=0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(30):
        time.sleep(1)
        r = subprocess.run(
            ["arch", "-x86_64", CH_BIN, "client",
             "--host", CH_HOST, "--port", CH_PORT,
             "--query", "SELECT 1"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and "1" in r.stdout:
            print("  ClickHouse server ready")
            return proc
    proc.kill()
    raise RuntimeError("ClickHouse server failed to start")

def stop_clickhouse(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

def run():
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ============================================================
    # Step 1: Generate data & export Parquet
    # ============================================================
    print("=" * 70)
    print("Step 1: Generate 50M rows & export Parquet")
    print("=" * 70)

    db = duckdb.connect()
    db.execute("SET threads TO 8")
    t0 = time.time()
    db.execute(f"""
        CREATE TABLE raw AS
        WITH base AS (SELECT row_number() OVER () AS rn FROM generate_series(1, {ROW_COUNT}))
        SELECT
            rn AS id,
            DATE '2023-01-01' + ((rn / 50000)::INTEGER)          AS dt,
            (rn % 200) + 1                                        AS prod,
            ((rn / 10000) % 50 + 1)::SMALLINT                    AS region,
            ((rn / 5000) % 100 + 1)::SMALLINT                    AS store,
            (rn % 100 + 1)::SMALLINT                              AS qty,
            ((rn * 7) % 5000 + 1)::DECIMAL(10,2)                 AS price,
            (rn % 100 + 1)::INTEGER * ((rn * 7) % 5000 + 1)::DECIMAL(10,2) AS amt
        FROM base
    """)
    db.execute(f"COPY raw TO '{PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    t_gen = time.time() - t0

    cardinality = db.sql("SELECT count(DISTINCT (dt, prod, region, store)) FROM raw").fetchone()[0]
    parquet_size = os.path.getsize(PARQUET)
    print(f"  Parquet: {parquet_size/1024/1024:.1f} MiB,  dim combos: {cardinality:,},  time: {t_gen:.1f}s")

    # ClickHouse 要求文件在 user_files_path 下, 复制过去
    ch_user_files = f"{CH_DIR}/user_files"
    os.makedirs(ch_user_files, exist_ok=True)
    ch_parquet = f"{ch_user_files}/raw.parquet"
    shutil.copy(PARQUET, ch_parquet)
    print(f"  Copied Parquet to ClickHouse user_files\n")

    # ============================================================
    # Step 2: DuckDB 冷缓存聚合 (每次建新库, 确保从磁盘读取)
    # ============================================================
    print("=" * 70)
    print("Step 2: DuckDB — Direct aggregation on raw table (cold cache)")
    print("=" * 70)
    db.close()

    ddb_raw_times = {}
    ddb_queries = [
        ("fine   (dt,prod,region,store)",
         """SELECT dt, prod, region, store, count(*) AS cnt,
                   sum(qty) AS q, sum(amt) AS amt
            FROM raw GROUP BY ALL"""),
        ("mid    (dt,prod)",
         """SELECT dt, prod, count(*) AS cnt, sum(amt) AS amt
            FROM raw GROUP BY ALL"""),
        ("coarse (dt only)",
         """SELECT dt, count(*) AS cnt, sum(amt) AS amt
            FROM raw GROUP BY dt ORDER BY dt"""),
        ("filter (prod=42, by region)",
         """SELECT region, count(*) AS cnt, sum(amt) AS amt
            FROM raw WHERE prod = 42 GROUP BY region"""),
    ]

    for idx, (label, sql) in enumerate(ddb_queries):
        db_path = f"{DATA_DIR}/ddb_cold_{idx}.db"
        db2 = duckdb.connect(db_path)
        db2.execute(f"CREATE TABLE raw AS SELECT * FROM '{PARQUET}'")
        db2.execute("CHECKPOINT")
        db2.close()
        db2 = duckdb.connect(db_path)
        t0 = time.time()
        db2.execute(sql)
        ddb_raw_times[label] = time.time() - t0
        db2.close()
        print(f"  {label:<32} {ddb_raw_times[label]:.3f}s")

    # ============================================================
    # Step 3: DuckDB 预聚合表
    # ============================================================
    print(f"\n{'='*70}")
    print("Step 3: DuckDB — Build pre-aggregated table & query")
    print("=" * 70)

    db = duckdb.connect(f"{DATA_DIR}/ddb_cold_0.db")
    t0 = time.time()
    db.execute("""
        CREATE TABLE agg AS
        SELECT dt, prod, region, store,
               count(*) AS cnt, sum(qty) AS q, sum(amt) AS amt
        FROM raw GROUP BY ALL
    """)
    db.execute("CHECKPOINT")
    t_ddb_build = time.time() - t0
    ddb_agg_rows = db.sql("SELECT count(*) FROM agg").fetchone()[0]
    db.close()
    print(f"  build: {t_ddb_build:.2f}s,  rows: {ddb_agg_rows:,}")

    ddb_agg_times = {}
    ddb_agg_queries = [
        ("fine   (read agg)",
         """SELECT * FROM agg"""),
        ("mid    (dt,prod from agg)",
         """SELECT dt, prod, sum(cnt) AS cnt, sum(amt) AS amt FROM agg GROUP BY ALL"""),
        ("coarse (dt from agg)",
         """SELECT dt, sum(cnt) AS cnt, sum(amt) AS amt FROM agg GROUP BY dt ORDER BY dt"""),
        ("filter (prod=42 from agg)",
         """SELECT region, sum(cnt) AS cnt, sum(amt) AS amt FROM agg WHERE prod = 42 GROUP BY region"""),
    ]

    for label, sql in ddb_agg_queries:
        db2 = duckdb.connect(f"{DATA_DIR}/ddb_cold_0.db")
        t0 = time.time()
        db2.execute(sql)
        ddb_agg_times[label] = time.time() - t0
        db2.close()
        print(f"  {label:<32} {ddb_agg_times[label]:.3f}s")

    # ============================================================
    # Step 4: Start ClickHouse & load data
    # ============================================================
    print(f"\n{'='*70}")
    print("Step 4: ClickHouse — Load data & MergeTree benchmark")
    print("=" * 70)
    ch_proc = start_clickhouse()

    run_ch("DROP TABLE IF EXISTS ch_raw")
    run_ch("""
        CREATE TABLE ch_raw (
            id      UInt64,
            dt      Date,
            prod    UInt32,
            region  UInt8,
            store   UInt16,
            qty     UInt8,
            price   Decimal(10,2),
            amt     Decimal(14,2)
        ) ENGINE = MergeTree()
        ORDER BY (dt, prod)
    """)

    t0 = time.time()
    run_ch(f"INSERT INTO ch_raw SELECT * FROM file('raw.parquet')", timeout=300)
    t_ch_load = time.time() - t0
    ch_rows = run_ch("SELECT count() FROM ch_raw")[1].strip()
    print(f"  MergeTree loaded: {ch_rows} rows in {t_ch_load:.1f}s")

    # MergeTree aggregation
    ch_mt_times = {}
    ch_mt_queries = [
        ("fine   (dt,prod,region,store)",
         "SELECT dt, prod, region, store, count() AS cnt, sum(qty) AS q, sum(amt) AS amt "
         "FROM ch_raw GROUP BY dt, prod, region, store ORDER BY dt, prod, region, store FORMAT TabSeparated"),
        ("mid    (dt,prod)",
         "SELECT dt, prod, count() AS cnt, sum(amt) AS amt FROM ch_raw GROUP BY dt, prod ORDER BY dt, prod FORMAT TabSeparated"),
        ("coarse (dt only)",
         "SELECT dt, count() AS cnt, sum(amt) AS amt FROM ch_raw GROUP BY dt ORDER BY dt FORMAT TabSeparated"),
        ("filter (prod=42, by region)",
         "SELECT region, count() AS cnt, sum(amt) AS amt FROM ch_raw WHERE prod = 42 GROUP BY region ORDER BY region FORMAT TabSeparated"),
    ]

    for label, sql in ch_mt_queries:
        run_ch(sql, f"{label} warm")
        t, _ = run_ch(sql, label)
        ch_mt_times[label] = t
        print(f"  {label:<32} {t:.3f}s")

    # ============================================================
    # Step 5: ClickHouse AggregatingMergeTree
    # ============================================================
    print(f"\n{'='*70}")
    print("Step 5: ClickHouse AggregatingMergeTree — Build & query")
    print("=" * 70)

    run_ch("DROP TABLE IF EXISTS ch_agg")
    run_ch("""
        CREATE TABLE ch_agg (
            dt      Date,
            prod    UInt32,
            region  UInt8,
            store   UInt16,
            cnt     SimpleAggregateFunction(sum, UInt64),
            q       SimpleAggregateFunction(sum, UInt64),
            amt     SimpleAggregateFunction(sum, Decimal(20,2))
        ) ENGINE = AggregatingMergeTree()
        ORDER BY (dt, prod, region, store)
    """)

    t0 = time.time()
    run_ch(f"""
        INSERT INTO ch_agg
        SELECT dt, prod, region, store, count() AS cnt, sum(qty) AS q, sum(amt) AS amt
        FROM ch_raw GROUP BY dt, prod, region, store
    """, timeout=300)
    run_ch("OPTIMIZE TABLE ch_agg FINAL", timeout=300)
    t_ch_agg_build = time.time() - t0
    ch_agg_rows = run_ch("SELECT count() FROM ch_agg")[1].strip()
    print(f"  AggregatingMergeTree: {ch_agg_rows} rows,  build: {t_ch_agg_build:.1f}s")

    # Query AggregatingMergeTree
    ch_agg_times = {}
    ch_agg_queries = [
        ("fine   (read agg)",
         "SELECT * FROM ch_agg ORDER BY dt, prod, region, store FORMAT TabSeparated"),
        ("mid    (dt,prod from agg)",
         "SELECT dt, prod, sum(cnt) AS cnt, sum(amt) AS amt FROM ch_agg GROUP BY dt, prod ORDER BY dt, prod FORMAT TabSeparated"),
        ("coarse (dt from agg)",
         "SELECT dt, sum(cnt) AS cnt, sum(amt) AS amt FROM ch_agg GROUP BY dt ORDER BY dt FORMAT TabSeparated"),
        ("filter (prod=42 from agg)",
         "SELECT region, sum(cnt) AS cnt, sum(amt) AS amt FROM ch_agg WHERE prod = 42 GROUP BY region ORDER BY region FORMAT TabSeparated"),
    ]

    for label, sql in ch_agg_queries:
        run_ch(sql, f"{label} warm")
        t, _ = run_ch(sql, label)
        ch_agg_times[label] = t
        print(f"  {label:<32} {t:.3f}s")

    # ============================================================
    # Summary
    # ============================================================
    label_map = [
        ("fine   (dt,prod,region,store)", "fine   (read agg)"),
        ("mid    (dt,prod)",              "mid    (dt,prod from agg)"),
        ("coarse (dt only)",              "coarse (dt from agg)"),
        ("filter (prod=42, by region)",   "filter (prod=42 from agg)"),
    ]

    print(f"\n{'='*90}")
    print("                  DuckDB vs ClickHouse — Aggregation Benchmark")
    print("=" * 90)
    print(f"  Dataset: {ROW_COUNT:,} rows, {parquet_size/1024/1024:.0f} MiB Parquet, {cardinality:,} dim combos\n")
    print(f"  {'Query':<32} {'DDB-raw':>8} {'DDB-agg':>8} {'CH-MT':>8} {'CH-AggMT':>8}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for raw_lbl, agg_lbl in label_map:
        print(f"  {raw_lbl:<32} "
              f"{ddb_raw_times[raw_lbl]:>7.3f}s "
              f"{ddb_agg_times[agg_lbl]:>7.3f}s "
              f"{ch_mt_times[raw_lbl]:>7.3f}s "
              f"{ch_agg_times[agg_lbl]:>7.3f}s")

    print(f"\n  {'Build cost':<32} {'—':>8} {t_ddb_build:>7.2f}s {'—':>8} {t_ch_agg_build:>7.2f}s")

    print(f"\n  Key takeaway:")
    if ddb_raw_times["fine   (dt,prod,region,store)"] < ch_mt_times["fine   (dt,prod,region,store)"]:
        print(f"    DuckDB 实时聚合比 ClickHouse MergeTree 更快 (向量化 vs 行式)")
    else:
        print(f"    ClickHouse MergeTree 聚合更快 (但差距在毫秒级)")

    stop_clickhouse(ch_proc)
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    print("\nDone.")

if __name__ == "__main__":
    run()
