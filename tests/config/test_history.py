"""Tests for the config-drift snapshot store."""
from __future__ import annotations

from pathlib import Path

import pytest

from rabbit_hunter.config.history import (
    HistoryEntry, diff, history, snapshot_if_changed,
)


def _write_config(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


CFG_V1 = "data:\n  symbols: [BTC-USDT-SWAP]\n"
CFG_V2 = "data:\n  symbols: [BTC-USDT-SWAP, ETH-USDT-SWAP]\n"
CFG_V3 = "data:\n  symbols: [BTC-USDT-SWAP, ETH-USDT-SWAP, SOL-USDT-SWAP]\n"


# ============================================================
# snapshot_if_changed
# ============================================================

def test_snapshot_creates_first_entry(tmp_path):
    cfg = tmp_path / "conf.yaml"
    _write_config(cfg, CFG_V1)
    entry = snapshot_if_changed(cfg, history_dir=tmp_path / "hist",
                                 now_utc="20260101-120000")
    assert entry is not None
    assert entry.source_path == str(cfg)
    assert (tmp_path / "hist" / "manifest.jsonl").exists()
    # The snapshot file exists with the same text
    assert Path(entry.snapshot_path).read_text() == CFG_V1


def test_snapshot_idempotent_no_change(tmp_path):
    """Second call with same content must return None and not write."""
    cfg = tmp_path / "conf.yaml"
    _write_config(cfg, CFG_V1)
    hist = tmp_path / "hist"
    snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-120000")
    result = snapshot_if_changed(cfg, history_dir=hist,
                                   now_utc="20260101-120001")
    assert result is None
    assert len(history(history_dir=hist)) == 1


def test_snapshot_captures_subsequent_changes(tmp_path):
    cfg = tmp_path / "conf.yaml"
    hist = tmp_path / "hist"
    _write_config(cfg, CFG_V1)
    e1 = snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000001")
    _write_config(cfg, CFG_V2)
    e2 = snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000002")
    _write_config(cfg, CFG_V3)
    e3 = snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000003")
    assert e1 and e2 and e3
    assert e1.config_hash != e2.config_hash != e3.config_hash
    assert len(history(history_dir=hist)) == 3


def test_snapshot_raises_when_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        snapshot_if_changed(tmp_path / "no.yaml",
                              history_dir=tmp_path / "hist")


def test_snapshot_records_note(tmp_path):
    cfg = tmp_path / "conf.yaml"
    _write_config(cfg, CFG_V1)
    entry = snapshot_if_changed(cfg, history_dir=tmp_path / "hist",
                                 note="backtest run 2026-01-01")
    assert entry.note == "backtest run 2026-01-01"


# ============================================================
# history filtering
# ============================================================

def test_history_filter_by_source(tmp_path):
    hist = tmp_path / "hist"
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    _write_config(cfg_a, CFG_V1)
    _write_config(cfg_b, CFG_V2)
    snapshot_if_changed(cfg_a, history_dir=hist, now_utc="20260101-000001")
    snapshot_if_changed(cfg_b, history_dir=hist, now_utc="20260101-000002")
    all_entries = history(history_dir=hist)
    assert len(all_entries) == 2
    only_a = history(history_dir=hist, source_path=cfg_a)
    assert len(only_a) == 1
    assert only_a[0].source_path == str(cfg_a)


def test_history_empty_when_no_manifest(tmp_path):
    assert history(history_dir=tmp_path / "no_history") == []


def test_history_survives_partial_manifest_writes(tmp_path):
    """A truncated JSONL line must not crash history reads."""
    hist = tmp_path / "hist"
    cfg = tmp_path / "c.yaml"
    _write_config(cfg, CFG_V1)
    snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000001")
    # Corrupt the manifest with a truncated line
    with (hist / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"timestamp_utc": "broken')   # missing closing brace + fields
    entries = history(history_dir=hist)
    assert len(entries) == 1   # good line intact


# ============================================================
# diff
# ============================================================

def test_diff_latest_vs_previous(tmp_path):
    cfg = tmp_path / "conf.yaml"
    hist = tmp_path / "hist"
    _write_config(cfg, CFG_V1)
    snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000001")
    _write_config(cfg, CFG_V2)
    snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000002")
    text = diff("previous", "latest", history_dir=hist)
    # V2 added ETH-USDT-SWAP so a "+" line must include it
    assert "+  symbols: [BTC-USDT-SWAP, ETH-USDT-SWAP]" in text
    # And the V1 line was removed
    assert "-  symbols: [BTC-USDT-SWAP]" in text


def test_diff_by_hash_prefix(tmp_path):
    cfg = tmp_path / "conf.yaml"
    hist = tmp_path / "hist"
    _write_config(cfg, CFG_V1)
    e1 = snapshot_if_changed(cfg, history_dir=hist,
                              now_utc="20260101-000001")
    _write_config(cfg, CFG_V2)
    e2 = snapshot_if_changed(cfg, history_dir=hist,
                              now_utc="20260101-000002")
    # 6-char prefix should be unique in a 2-entry manifest
    text = diff(e1.config_hash[:6], e2.config_hash[:6], history_dir=hist)
    assert text  # non-empty


def test_diff_by_integer_index(tmp_path):
    cfg = tmp_path / "conf.yaml"
    hist = tmp_path / "hist"
    _write_config(cfg, CFG_V1)
    snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000001")
    _write_config(cfg, CFG_V2)
    snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000002")
    text = diff("0", "1", history_dir=hist)
    assert text


def test_diff_identical_revisions_empty(tmp_path):
    cfg = tmp_path / "conf.yaml"
    hist = tmp_path / "hist"
    _write_config(cfg, CFG_V1)
    snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000001")
    text = diff("latest", "latest", history_dir=hist)
    assert text == ""


def test_diff_previous_without_two_snapshots_raises(tmp_path):
    cfg = tmp_path / "conf.yaml"
    hist = tmp_path / "hist"
    _write_config(cfg, CFG_V1)
    snapshot_if_changed(cfg, history_dir=hist, now_utc="20260101-000001")
    with pytest.raises(ValueError):
        diff("previous", "latest", history_dir=hist)


def test_diff_ambiguous_hash_prefix_raises(tmp_path):
    """If two snapshots share the same prefix, refuse to guess."""
    hist = tmp_path / "hist"
    # Force two entries with same 3-char prefix via a manual manifest.
    # In practice, hash collisions on 16-char SHA-256 prefix are astronomically
    # rare, but ambiguous SHORT prefixes could happen — test the guard.
    hist.mkdir()
    (hist / "aa1111.yaml").write_text("v1", encoding="utf-8")
    (hist / "aa2222.yaml").write_text("v2", encoding="utf-8")
    from rabbit_hunter.config.history import _append_manifest
    _append_manifest(hist, HistoryEntry(
        timestamp_utc="2026-01-01T00:00:01",
        config_hash="aa1111abcdef1234",
        source_path="c.yaml", snapshot_path=str(hist / "aa1111.yaml"),
    ))
    _append_manifest(hist, HistoryEntry(
        timestamp_utc="2026-01-01T00:00:02",
        config_hash="aa2222abcdef5678",
        source_path="c.yaml", snapshot_path=str(hist / "aa2222.yaml"),
    ))
    with pytest.raises(ValueError):
        diff("aa", "latest", history_dir=hist)
