import json
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from config import DB_PATH, PEOPLE_DIR, TOP_K, ensure_dirs
from face_recognition_engine import cosine_sim, l2_normalize


@dataclass
class PersonRecord:
    name: str
    folder: str
    centroid: List[float]
    num_samples: int


def sanitize_name(name: str) -> str:
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch in ["_", "-"]:
            out.append(ch)
        elif 0xAC00 <= ord(ch) <= 0xD7A3:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "unknown"


class FaceDatabase:
    def __init__(self, db_path: str = DB_PATH, people_dir: str = PEOPLE_DIR):
        self.db_path = db_path
        self.people_dir = people_dir
        ensure_dirs()
        self.people: List[PersonRecord] = []
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.db_path):
            self.people = []
            return

        with open(self.db_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.people = [PersonRecord(**item) for item in raw.get("people", [])]

    def save(self) -> None:
        payload = {"people": [p.__dict__ for p in self.people]}
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def upsert_person(self, name: str, embeddings: List[np.ndarray], crops: List[np.ndarray]) -> None:
        folder = sanitize_name(name)
        person_dir = os.path.join(self.people_dir, folder)

        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        os.makedirs(person_dir, exist_ok=True)

        for idx, crop in enumerate(crops, start=1):
            if crop is not None and crop.size > 0:
                cv2.imwrite(os.path.join(person_dir, f"{idx:03d}.jpg"), crop)

        centroid = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0)).astype(np.float32)

        self.people = [p for p in self.people if p.name != name]
        self.people.append(
            PersonRecord(
                name=name,
                folder=folder,
                centroid=centroid.tolist(),
                num_samples=len(embeddings),
            )
        )
        self.save()

    def rank(self, emb: np.ndarray, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        if not self.people:
            return []

        scored = []
        for person in self.people:
            score = cosine_sim(np.asarray(person.centroid, dtype=np.float32), emb)
            scored.append((person.name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
