from pathlib import Path
import pandas as pd
from rabbit_hunter.observability.snapshot import SnapshotWriter


def test_snapshot_writer_partitions_by_day(tmp_path):
    w = SnapshotWriter(root=tmp_path)
    records = [
        {"timestamp": 1_700_000_000_000, "symbol": "BTC-USDT-SWAP", "action": "wait",
         "conviction": 0.2, "long_score": {}, "order_placed": False},
        {"timestamp": 1_700_086_400_000, "symbol": "BTC-USDT-SWAP", "action": "open_long",
         "conviction": 0.7, "long_score": {"trend_following": 0.7}, "order_placed": True},
    ]
    w.append(records)
    paths = w.flush()
    assert len(paths) == 2  # 两天两文件
    df = pd.concat([pd.read_parquet(p) for p in paths])
    assert len(df) == 2
    assert set(df["action"]) == {"wait", "open_long"}


def test_snapshot_writer_no_records_no_files(tmp_path):
    w = SnapshotWriter(root=tmp_path)
    paths = w.flush()
    assert paths == []
