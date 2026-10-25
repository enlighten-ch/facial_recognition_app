#!/usr/bin/env python3
"""
Export structured face DB (SQLite metadata + BIN embeddings) into a readable CSV.

Default input files:
  - face_app_data/face_meta.db
  - face_app_data/face_embeddings.bin
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import struct
import sys
from pathlib import Path


TABLE_NAME = "people"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "face_app_data"
ㅇ

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export face_meta.db + face_embeddings.bin to CSV.",
    )
    p.add_argument(
        "--meta-db",
        default=str(DEFAULT_DATA_DIR / "face_meta.db"),
        help="Path to face_meta.db",
    )
    p.add_argument(
        "--embeddings-bin",
        default=str(DEFAULT_DATA_DIR / "face_embeddings.bin"),
        help="Path to face_embeddings.bin",
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_DATA_DIR / "face_structured_export_metadata.csv"),
        help="Output CSV file path",
    )
    p.add_argument(
        "--metadata-only",
        action="store_true",
        help="Export metadata only, without embeddings_json column (kept for backward compatibility).",
    )
    p.add_argument(
        "--with-embeddings-json",
        action="store_true",
        help="Include full embeddings_json column (very large file, some editors may fail to open).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Export first N people only (0 means all)",
    )
    return p.parse_args()


def read_people_rows(meta_db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(meta_db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            f"""
            SELECT
                name, department, offset_bytes, length_bytes,
                embedding_count, embedding_dim, hash, updated_at
            FROM {TABLE_NAME}
            ORDER BY name ASC
            """,
        ).fetchall()
    finally:
        conn.close()


def decode_embeddings(raw: bytes, offset: int, length: int, count: int, dim: int) -> list[list[float]]:
    if offset < 0 or length < 0 or count <= 0 or dim <= 0:
        return []
    if offset + length > len(raw):
        return []

    out: list[list[float]] = []
    p = offset
    limit = offset + length
    bytes_per_embedding = dim * 4

    for _ in range(count):
        if p + bytes_per_embedding > limit:
            break
        vec = list(struct.unpack_from("<" + ("f" * dim), raw, p))
        out.append(vec)
        p += bytes_per_embedding
    return out


def main() -> int:
    args = parse_args()
    meta_db_path = Path(args.meta_db).expanduser().resolve()
    embeddings_bin_path = Path(args.embeddings_bin).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not meta_db_path.exists():
        print(f"[ERROR] meta db not found: {meta_db_path}", file=sys.stderr)
        return 1
    if not embeddings_bin_path.exists():
        print(f"[ERROR] embeddings bin not found: {embeddings_bin_path}", file=sys.stderr)
        return 1

    try:
        rows = read_people_rows(meta_db_path)
    except Exception as exc:
        print(f"[ERROR] failed to read sqlite rows: {exc}", file=sys.stderr)
        return 1

    try:
        raw_bin = embeddings_bin_path.read_bytes()
    except Exception as exc:
        print(f"[ERROR] failed to read embeddings bin: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_only = args.metadata_only or (not args.with_embeddings_json)

    fields = [
        "name",
        "department",
        "embedding_count",
        "embedding_dim",
        "updated_at",
        "hash",
        "offset_bytes",
        "length_bytes",
    ]
    if not metadata_only:
        fields.append("embeddings_json")

    try:
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            exported = 0
            for row in rows:
                if args.limit > 0 and exported >= args.limit:
                    break
                name = str(row["name"] or "").strip()
                department = str(row["department"] or "").strip()
                offset = int(row["offset_bytes"] or 0)
                length = int(row["length_bytes"] or 0)
                count = int(row["embedding_count"] or 0)
                dim = int(row["embedding_dim"] or 0)
                hash_hex = str(row["hash"] or "").strip()
                updated_at = int(row["updated_at"] or 0)

                record = {
                    "name": name,
                    "department": department,
                    "embedding_count": count,
                    "embedding_dim": dim,
                    "updated_at": updated_at,
                    "hash": hash_hex,
                    "offset_bytes": offset,
                    "length_bytes": length,
                }

                if not metadata_only:
                    embeddings = decode_embeddings(raw_bin, offset, length, count, dim)
                    record["embeddings_json"] = json.dumps(embeddings, ensure_ascii=False)

                writer.writerow(record)
                exported += 1
    except Exception as exc:
        print(f"[ERROR] failed to write csv: {exc}", file=sys.stderr)
        return 1

    print("[OK] export complete")
    print(f"  meta_db        : {meta_db_path}")
    print(f"  embeddings_bin : {embeddings_bin_path}")
    print(f"  output_csv     : {output_path}")
    print(f"  people_total   : {len(rows)}")
    print(f"  people_exported: {exported if args.limit > 0 else len(rows)}")
    print(f"  mode           : {'metadata-only' if metadata_only else 'full-with-embeddings-json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
