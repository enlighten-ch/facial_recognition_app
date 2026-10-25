import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    DB_IMAGE_EXTENSIONS,
    DB_IMAGE_FILENAME_PATTERN,
    DB_JSON_INDENT,
    DB_PATH,
    MAX_EMBEDDINGS_PER_PERSON,
    PEOPLE_DIR,
    SANITIZE_UNKNOWN_NAME,
    TOP_K,
    ensure_dirs,
)
from face_recognition_engine import l2_normalize


@dataclass
class PersonRecord:
    name: str
    folder: str
    centroid: List[float]
    num_samples: int
    embeddings: Optional[List[List[float]]] = None


def sanitize_name(name: str) -> str:
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch in ["_", "-"]:
            out.append(ch)
        elif 0xAC00 <= ord(ch) <= 0xD7A3:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or SANITIZE_UNKNOWN_NAME


class FaceDatabase:
    def __init__(self, db_path: str = DB_PATH, people_dir: str = PEOPLE_DIR):
        self.db_path = db_path
        self.people_dir = people_dir
        ensure_dirs()
        self.people: List[PersonRecord] = []
        self._centroid_matrix = np.empty((0, 0), dtype=np.float32)
        self._names: List[str] = []
        self._next_sample_idx_by_name: Dict[str, int] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.db_path):
            self.people = []
            self._rebuild_rank_cache()
            self._rebuild_sample_index_cache()
            return

        with open(self.db_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.people = [PersonRecord(**item) for item in raw.get("people", [])]
        self._normalize_people_schema()
        self._rebuild_rank_cache()
        self._rebuild_sample_index_cache()

    def save(self) -> None:
        payload = {"people": [p.__dict__ for p in self.people]}
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=DB_JSON_INDENT)

    def upsert_person(self, name: str, embeddings: List[np.ndarray], crops: List[np.ndarray]) -> None:
        folder = sanitize_name(name)
        person_dir = os.path.join(self.people_dir, folder)

        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        os.makedirs(person_dir, exist_ok=True)

        embs = [np.asarray(e, dtype=np.float32) for e in embeddings]
        embs = self._prune_embeddings_to_limit(embs, MAX_EMBEDDINGS_PER_PERSON)
        kept_count = len(embs)

        for idx, crop in enumerate(crops[:kept_count], start=1):
            if crop is not None and crop.size > 0:
                cv2.imwrite(os.path.join(person_dir, DB_IMAGE_FILENAME_PATTERN.format(idx=idx)), crop)

        centroid = l2_normalize(np.mean(np.stack(embs, axis=0), axis=0)).astype(np.float32)

        self.people = [p for p in self.people if p.name != name]
        self.people.append(
            PersonRecord(
                name=name,
                folder=folder,
                centroid=centroid.tolist(),
                num_samples=kept_count,
                embeddings=[e.tolist() for e in embs],
            )
        )
        self._rebuild_rank_cache()
        self._next_sample_idx_by_name[name] = kept_count + 1
        self.save()

    def rank(self, emb: np.ndarray, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        if len(self._names) == 0:
            return []

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if self._centroid_matrix.shape[1] != emb.shape[0]:
            return []

        sims = self._centroid_matrix @ emb
        k = max(1, min(top_k, sims.shape[0]))

        if k == sims.shape[0]:
            idxs = np.argsort(-sims)
        else:
            idxs = np.argpartition(-sims, k - 1)[:k]
            idxs = idxs[np.argsort(-sims[idxs])]

        return [(self._names[int(i)], float(sims[int(i)])) for i in idxs]

    def append_sample_to_person(self, name: str, emb: np.ndarray, crop: np.ndarray) -> bool:
        target = None
        for person in self.people:
            if person.name == name:
                target = person
                break

        if target is None:
            return False

        existing_embs = target.embeddings or [target.centroid]
        embs = [np.asarray(e, dtype=np.float32) for e in existing_embs]
        embs.append(np.asarray(emb, dtype=np.float32))
        embs = self._prune_embeddings_to_limit(embs, MAX_EMBEDDINGS_PER_PERSON)

        new_centroid = l2_normalize(np.mean(np.stack(embs, axis=0), axis=0)).astype(np.float32)
        target.centroid = new_centroid.tolist()
        target.num_samples = len(embs)
        target.embeddings = [e.tolist() for e in embs]
        self._rebuild_rank_cache()

        person_dir = os.path.join(self.people_dir, target.folder)
        os.makedirs(person_dir, exist_ok=True)
        if crop is not None and crop.size > 0:
            next_idx = self._next_sample_idx_by_name.get(name)
            if next_idx is None:
                next_idx = self._scan_next_sample_index(person_dir)
            cv2.imwrite(os.path.join(person_dir, DB_IMAGE_FILENAME_PATTERN.format(idx=next_idx)), crop)
            self._next_sample_idx_by_name[name] = next_idx + 1

        self.save()
        return True

    def _normalize_people_schema(self) -> None:
        for person in self.people:
            if not person.embeddings:
                person.embeddings = [person.centroid]
            emb_list = self._prune_embeddings_to_limit(
                [np.asarray(e, dtype=np.float32) for e in person.embeddings],
                MAX_EMBEDDINGS_PER_PERSON,
            )
            person.centroid = l2_normalize(
                np.mean(np.stack(emb_list, axis=0), axis=0)
            ).astype(np.float32).tolist()
            person.num_samples = len(emb_list)
            person.embeddings = [e.tolist() for e in emb_list]

    def _prune_embeddings_to_limit(self, embs: List[np.ndarray], limit: int) -> List[np.ndarray]:
        if not embs:
            return embs

        lim = max(1, int(limit))
        out = [np.asarray(e, dtype=np.float32) for e in embs]

        # Remove low-similarity embeddings one by one until within limit.
        while len(out) > lim:
            centroid = l2_normalize(np.mean(np.stack(out, axis=0), axis=0)).astype(np.float32)
            sims = [float(np.dot(e, centroid)) for e in out]
            drop_idx = int(np.argmin(sims))
            out.pop(drop_idx)
        return out

    def _rebuild_rank_cache(self) -> None:
        self._names = [p.name for p in self.people]
        if not self.people:
            self._centroid_matrix = np.empty((0, 0), dtype=np.float32)
            return
        self._centroid_matrix = np.asarray([p.centroid for p in self.people], dtype=np.float32)

    def _scan_next_sample_index(self, person_dir: str) -> int:
        if not os.path.isdir(person_dir):
            return 1
        count = sum(1 for fn in os.listdir(person_dir) if fn.lower().endswith(DB_IMAGE_EXTENSIONS))
        return count + 1

    def _rebuild_sample_index_cache(self) -> None:
        self._next_sample_idx_by_name = {}
        for person in self.people:
            person_dir = os.path.join(self.people_dir, person.folder)
            self._next_sample_idx_by_name[person.name] = self._scan_next_sample_index(person_dir)
