"""Config drift tracker — every non-trivial config change is snapshotted
so weeks later you can answer "why did trades start losing on 2026-05-14?".

Two mechanisms:

  1. Passive snapshots — `snapshot_if_changed(config_path, history_dir)`
     computes a hash of the current config file. If different from the
     last recorded snapshot, writes a new one to
     `configs/.history/<timestamp>-<hash>.yaml` and appends a manifest
     entry.

     Callers wire this into backtest and shadow-run entry points so
     every substantive run automatically stamps whatever config was
     loaded. No manual bookkeeping.

  2. Explicit diff — `diff(rev_a, rev_b)` renders a text diff between
     any two recorded snapshots. `history()` lists what's on file.

The snapshot store is one directory of YAML files + a JSONL manifest —
same principle as models/retrain_log.jsonl. Human-readable, git-diffable,
no database.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_HISTORY_DIR = Path("configs/.history")
MANIFEST_NAME = "manifest.jsonl"


@dataclass
class HistoryEntry:
    """One recorded snapshot's metadata."""
    timestamp_utc: str
    config_hash: str
    source_path: str
    snapshot_path: str
    note: str = ""


# ============================================================
# Hashing + IO
# ============================================================

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _manifest_path(history_dir: Path) -> Path:
    return history_dir / MANIFEST_NAME


def _load_manifest(history_dir: Path) -> list[HistoryEntry]:
    p = _manifest_path(history_dir)
    if not p.exists():
        return []
    out: list[HistoryEntry] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            out.append(HistoryEntry(
                timestamp_utc=data["timestamp_utc"],
                config_hash=data["config_hash"],
                source_path=data["source_path"],
                snapshot_path=data["snapshot_path"],
                note=data.get("note", ""),
            ))
        except (json.JSONDecodeError, KeyError):
            # Skip broken lines rather than aborting — a truncated
            # write shouldn't kill history reads.
            continue
    return out


def _append_manifest(history_dir: Path, entry: HistoryEntry) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    p = _manifest_path(history_dir)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp_utc": entry.timestamp_utc,
            "config_hash": entry.config_hash,
            "source_path": entry.source_path,
            "snapshot_path": entry.snapshot_path,
            "note": entry.note,
        }) + "\n")


# ============================================================
# Snapshot logic
# ============================================================

def snapshot_if_changed(
    config_path: Path,
    history_dir: Path | None = None,
    note: str = "",
    now_utc: str | None = None,
) -> HistoryEntry | None:
    """Take a snapshot of `config_path` if its hash differs from the
    latest recorded snapshot. Returns the new entry on save, or None if
    the file was unchanged (idempotent — safe to call every backtest/tick).
    """
    history_dir = history_dir or DEFAULT_HISTORY_DIR
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    text = _read_text(config_path)
    hash_now = _hash_text(text)
    entries = _load_manifest(history_dir)
    latest_for_source = [e for e in entries if e.source_path == str(config_path)]
    if latest_for_source and latest_for_source[-1].config_hash == hash_now:
        return None
    ts = now_utc or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    snapshot_name = f"{ts}-{hash_now}.yaml"
    snapshot_path = history_dir / snapshot_name
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(text, encoding="utf-8")
    entry = HistoryEntry(
        timestamp_utc=(now_utc or
                        datetime.now(timezone.utc).isoformat(timespec="seconds")),
        config_hash=hash_now,
        source_path=str(config_path),
        snapshot_path=str(snapshot_path),
        note=note,
    )
    _append_manifest(history_dir, entry)
    return entry


def history(
    history_dir: Path | None = None,
    source_path: Path | None = None,
) -> list[HistoryEntry]:
    """Return all recorded snapshots, optionally filtered by source config."""
    history_dir = history_dir or DEFAULT_HISTORY_DIR
    entries = _load_manifest(history_dir)
    if source_path is not None:
        entries = [e for e in entries if e.source_path == str(source_path)]
    return entries


def _resolve_revision(entries: list[HistoryEntry], rev: str) -> HistoryEntry:
    """Resolve a revision spec: `latest`, `previous`, hash prefix, or index."""
    if not entries:
        raise ValueError("no history entries")
    if rev == "latest":
        return entries[-1]
    if rev == "previous":
        if len(entries) < 2:
            raise ValueError("no previous revision (only one snapshot)")
        return entries[-2]
    # Try hash-prefix match
    matches = [e for e in entries if e.config_hash.startswith(rev)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous hash prefix {rev!r}: matches {[m.config_hash for m in matches]}"
        )
    # Try index
    try:
        idx = int(rev)
        return entries[idx]
    except (ValueError, IndexError):
        pass
    raise ValueError(f"could not resolve revision {rev!r}")


def diff(
    rev_a: str,
    rev_b: str,
    history_dir: Path | None = None,
    source_path: Path | None = None,
) -> str:
    """Unified diff between two recorded snapshots.

    `rev_a` and `rev_b` accept "latest", "previous", a hash prefix, or
    an integer index into the manifest. Returns the unified-diff text
    (empty string when identical)."""
    history_dir = history_dir or DEFAULT_HISTORY_DIR
    entries = history(history_dir=history_dir, source_path=source_path)
    entry_a = _resolve_revision(entries, rev_a)
    entry_b = _resolve_revision(entries, rev_b)
    text_a = _read_text(Path(entry_a.snapshot_path))
    text_b = _read_text(Path(entry_b.snapshot_path))
    lines = difflib.unified_diff(
        text_a.splitlines(keepends=True),
        text_b.splitlines(keepends=True),
        fromfile=f"{entry_a.config_hash} ({entry_a.timestamp_utc})",
        tofile=f"{entry_b.config_hash} ({entry_b.timestamp_utc})",
        n=3,
    )
    return "".join(lines)
