#!/usr/bin/env python3
"""
Convert legacy JSON face DB into structured DB files used by Android apps:
1) CSV snapshot (user-visible metadata only): face_db.csv
2) Metadata SQLite: face_meta.db
3) Embeddings binary: face_embeddings.bin

Usage:
  python3 scripts/convert_face_db.py \
    --input face_app_data/face_db.json \
    --output-dir face_app_data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TABLE_NAME = "people"


@dataclass
class PersonRow:
    name: str
    department: str
    embeddings: list[list[float]]
    hash_hex: str
    updated_at: int


def update_digest_int(digest: "hashlib._Hash", value: int) -> None:
    digest.update(bytes(((value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)))


def float_to_int_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def compute_hash(name: str, department: str, embeddings: list[list[float]]) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(department.encode("utf-8"))
    digest.update(b"\x00")
    update_digest_int(digest, len(embeddings))
    for emb in embeddings:
        update_digest_int(digest, len(emb))
        for v in emb:
            update_digest_int(digest, float_to_int_bits(v))
    return digest.hexdigest()


def parse_root(raw_obj: Any) -> list[dict[str, Any]]:
    if isinstance(raw_obj, list):
        return [x for x in raw_obj if isinstance(x, dict)]
    if not isinstance(raw_obj, dict):
        return []

    for key in ("people", "persons", "faces"):
        val = raw_obj.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
        if isinstance(val, dict):
            out: list[dict[str, Any]] = []
            for k, item in val.items():
                if not isinstance(item, dict):
                    continue
                if not item.get("name"):
                    item = dict(item)
                    item["name"] = str(k)
                out.append(item)
            return out
    return []


def parse_embeddings(person: dict[str, Any]) -> list[list[float]]:
    def to_vec_list(raw: Any) -> list[list[float]]:
        if not isinstance(raw, list):
            return []
        out: list[list[float]] = []
        for item in raw:
            if not isinstance(item, list):
                continue
            vec: list[float] = []
            for v in item:
                try:
                    vec.append(float(v))
                except Exception:
                    vec.append(0.0)
            if vec:
                out.append(vec)
        return out

    embeddings = to_vec_list(person.get("embeddings"))
    if embeddings:
        return embeddings

    by_direction = person.get("embeddings_by_direction")
    if isinstance(by_direction, dict):
        merged: list[list[float]] = []
        for key in sorted(by_direction.keys()):
            merged.extend(to_vec_list(by_direction.get(key)))
        if merged:
            return merged

    samples = to_vec_list(person.get("samples"))
    if samples:
        return samples

    return []


def normalize_embeddings(embeddings: list[list[float]]) -> list[list[float]]:
    if not embeddings:
        return []
    dim = len(embeddings[0])
    if dim <= 0:
        return []
    normalized: list[list[float]] = []
    for emb in embeddings:
        vec = [float(emb[i]) if i < len(emb) else 0.0 for i in range(dim)]
        normalized.append(vec)
    return normalized


def build_rows(raw_people: list[dict[str, Any]]) -> tuple[list[PersonRow], int]:
    rows: list[PersonRow] = []
    skipped = 0
    now_ms = int(time.time() * 1000)

    for person in raw_people:
        name = str(person.get("name", "")).strip()
        if not name:
            skipped += 1
            continue
        department = str(person.get("department", person.get("folder", ""))).strip()
        embeddings = normalize_embeddings(parse_embeddings(person))
        if not embeddings:
            skipped += 1
            continue
        updated_at = person.get("updatedAt", person.get("updated_at", now_ms))
        try:
            updated_at = int(updated_at)
        except Exception:
            updated_at = now_ms

        hash_hex = str(person.get("hash", "")).strip() or compute_hash(name, department, embeddings)
        rows.append(
            PersonRow(
                name=name,
                department=department,
                embeddings=embeddings,
                hash_hex=hash_hex,
                updated_at=updated_at,
            ),
        )

    rows.sort(key=lambda x: x.name)
    return rows, skipped


def write_csv(rows: list[PersonRow], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "department", "embedding_count", "updated_at"])
        for row in rows:
            writer.writerow(
                [
                    row.name,
                    row.department,
                    len(row.embeddings),
                    row.updated_at,
                ],
            )


def write_bin_and_sqlite(rows: list[PersonRow], bin_path: Path, sqlite_path: Path) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    meta_rows: list[tuple[str, str, int, int, int, int, str, int]] = []
    with bin_path.open("wb") as bf:
        for row in rows:
            count = len(row.embeddings)
            dim = len(row.embeddings[0]) if count > 0 else 0
            start = offset
            for emb in row.embeddings:
                for i in range(dim):
                    v = float(emb[i]) if i < len(emb) else 0.0
                    bf.write(struct.pack("<f", v))
                    offset += 4
            length = offset - start
            meta_rows.append((row.name, row.department, start, length, count, dim, row.hash_hex, row.updated_at))

    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME}(
                name TEXT PRIMARY KEY,
                department TEXT,
                offset_bytes INTEGER,
                length_bytes INTEGER,
                embedding_count INTEGER,
                embedding_dim INTEGER,
                hash TEXT,
                updated_at INTEGER
            )
            """,
        )
        conn.execute(f"DELETE FROM {TABLE_NAME}")
        conn.executemany(
            f"""
            INSERT INTO {TABLE_NAME}
            (name, department, offset_bytes, length_bytes, embedding_count, embedding_dim, hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            meta_rows,
        )
        conn.commit()
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert JSON face DB to CSV + SQLite + BIN.")
    p.add_argument("--input", required=True, help="Input JSON DB file path.")
    p.add_argument("--output-dir", required=True, help="Output directory.")
    p.add_argument("--csv-name", default="face_db.csv", help="Output CSV file name.")
    p.add_argument("--meta-db-name", default="face_meta.db", help="Output SQLite metadata DB file name.")
    p.add_argument("--bin-name", default="face_embeddings.bin", help="Output embeddings binary file name.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    csv_path = output_dir / args.csv_name
    meta_db_path = output_dir / args.meta_db_name
    bin_path = output_dir / args.bin_name

    if not input_path.exists():
        print(f"[ERROR] input not found: {input_path}", file=sys.stderr)
        return 1

    try:
        raw_obj = json.loads(input_path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        print(f"[ERROR] failed to parse JSON: {e}", file=sys.stderr)
        return 1

    raw_people = parse_root(raw_obj)
    rows, skipped = build_rows(raw_people)
    if not rows:
        print("[ERROR] no valid person rows found in input.", file=sys.stderr)
        return 1

    write_csv(rows, csv_path)
    write_bin_and_sqlite(rows, bin_path, meta_db_path)

    total_embeddings = sum(len(r.embeddings) for r in rows)
    print("[OK] conversion completed")
    print(f"  input_json      : {input_path}")
    print(f"  output_csv      : {csv_path}")
    print(f"  output_meta_db  : {meta_db_path}")
    print(f"  output_bin      : {bin_path}")
    print(f"  people          : {len(rows)}")
    print(f"  embeddings      : {total_embeddings}")
    print(f"  skipped_records : {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
