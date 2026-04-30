from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np


DEFAULT_RECORD_ROOT = Path(
    os.getenv("TP2_SESSION_RECORD_DIR", "/srv/tp2/frames/autonomous")
).expanduser()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def relabel_key(frame_seq: int, label_index: int) -> str:
    return f"{int(frame_seq)}:{int(label_index)}"


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def is_session_dir(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.jsonl").exists()


SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")


def safe_entry_name(value: str, *, suffix: str | None = None) -> str:
    name = Path(str(value).strip()).name
    if suffix and not name.lower().endswith(suffix.lower()):
        name += suffix
    if not SAFE_NAME_RE.fullmatch(name):
        raise ValueError("invalid safe name")
    return name


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


@dataclass(frozen=True)
class SessionSummary:
    id: str
    path: str
    frames: int
    critical: int
    reviewed: int
    images: int
    has_video: bool
    started_at: str | None
    review_status: str | None
    tags: list[str]
    modified_at: str


class SessionCatalog:
    def __init__(self, record_root: Path, *, initial_session_id: str | None = None) -> None:
        root = record_root.expanduser().resolve()
        if is_session_dir(root):
            self.record_root = root.parent
            self.initial_session_id = initial_session_id or root.name
        else:
            self.record_root = root
            self.initial_session_id = initial_session_id
        self.lock = threading.RLock()

    def sessions(self) -> list[SessionSummary]:
        if not self.record_root.exists():
            return []
        summaries: list[SessionSummary] = []
        for child in self.record_root.iterdir():
            if not is_session_dir(child):
                continue
            summaries.append(self._summarize(child))
        summaries.sort(key=lambda item: item.modified_at, reverse=True)
        return summaries

    def latest_session_id(self) -> str | None:
        sessions = self.sessions()
        if not sessions:
            return None
        if self.initial_session_id and any(item.id == self.initial_session_id for item in sessions):
            return self.initial_session_id
        return sessions[0].id

    def load(self, session_id: str | None = None) -> tuple[str, "SessionData"]:
        selected_id = self.resolve_session_id(session_id)
        if selected_id is None:
            raise FileNotFoundError(f"No session directories under {self.record_root}")
        path = self.session_path(selected_id)
        if not is_session_dir(path):
            raise FileNotFoundError(f"No manifest.jsonl for session {selected_id}")
        return selected_id, SessionData.load(path)

    def resolve_session_id(self, session_id: str | None) -> str | None:
        cleaned = unquote(session_id or "").strip()
        if cleaned:
            path = self.session_path(cleaned)
            if is_session_dir(path):
                return path.name
        return self.latest_session_id()

    def session_path(self, session_id: str) -> Path:
        candidate = (self.record_root / Path(session_id).name).resolve()
        if not path_is_relative_to(candidate, self.record_root):
            raise ValueError("session id escapes record root")
        return candidate

    def rename(self, session_id: str, new_id: str) -> SessionSummary:
        old_path = self.session_path(session_id)
        if not is_session_dir(old_path):
            raise FileNotFoundError(f"No manifest.jsonl for session {session_id}")
        new_name = safe_entry_name(new_id)
        new_path = (self.record_root / new_name).resolve()
        if not path_is_relative_to(new_path, self.record_root):
            raise ValueError("new session id escapes record root")
        if new_path.exists():
            raise FileExistsError(f"session already exists: {new_name}")
        old_path.rename(new_path)
        self.initial_session_id = new_name
        return self._summarize(new_path)

    def _summarize(self, path: Path) -> SessionSummary:
        manifest = read_jsonl(path / "manifest.jsonl")
        critical = sum(1 for item in manifest if (item.get("critical") or {}).get("is_critical"))
        reviews = 0
        reviews_path = path / "labels_reviewed.json"
        if reviews_path.exists():
            try:
                payload = json.loads(reviews_path.read_text(encoding="utf-8"))
                raw = payload.get("reviews", {})
                reviews = len(raw) if isinstance(raw, dict) else 0
            except json.JSONDecodeError:
                reviews = 0
        images_dir = path / "images"
        images = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
        started_at = None
        review_status = None
        tags: list[str] = []
        session_meta = path / "session.json"
        if session_meta.exists():
            try:
                payload = json.loads(session_meta.read_text(encoding="utf-8"))
                started_at = payload.get("started_at")
                review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
                review_status = review.get("status")
                raw_tags = review.get("tags")
                tags = [str(item) for item in raw_tags] if isinstance(raw_tags, list) else []
            except json.JSONDecodeError:
                started_at = None
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        return SessionSummary(
            id=path.name,
            path=str(path),
            frames=len(manifest),
            critical=critical,
            reviewed=reviews,
            images=images,
            has_video=(path / "session.mp4").exists(),
            started_at=started_at,
            review_status=review_status,
            tags=tags,
            modified_at=mtime,
        )


@dataclass
class SessionData:
    root: Path
    manifest: list[dict[str, Any]]
    reviews: dict[str, dict[str, Any]]
    manual_labels: dict[str, list[dict[str, Any]]]
    lock: threading.RLock

    @classmethod
    def load(cls, root: Path) -> "SessionData":
        root = root.expanduser().resolve()
        manifest = read_jsonl(root / "manifest.jsonl")
        reviews_path = root / "labels_reviewed.json"
        reviews: dict[str, dict[str, Any]] = {}
        if reviews_path.exists():
            try:
                payload = json.loads(reviews_path.read_text(encoding="utf-8"))
                raw_reviews = payload.get("reviews", {})
                if isinstance(raw_reviews, dict):
                    reviews = {
                        str(key): value
                        for key, value in raw_reviews.items()
                        if isinstance(value, dict)
                    }
            except json.JSONDecodeError:
                reviews = {}
        manual_path = root / "manual_labels.json"
        manual_labels: dict[str, list[dict[str, Any]]] = {}
        if manual_path.exists():
            try:
                payload = json.loads(manual_path.read_text(encoding="utf-8"))
                raw = payload.get("labels", {})
                if isinstance(raw, dict):
                    for key, items in raw.items():
                        if not isinstance(items, list):
                            continue
                        cleaned = [item for item in items if isinstance(item, dict) and item.get("id")]
                        if cleaned:
                            manual_labels[str(key)] = cleaned
            except json.JSONDecodeError:
                manual_labels = {}
        return cls(
            root=root,
            manifest=manifest,
            reviews=reviews,
            manual_labels=manual_labels,
            lock=threading.RLock(),
        )

    def classes(self) -> list[str]:
        values: set[str] = set()
        for item in self.manifest:
            for label in item.get("labels", []) or []:
                label_class = label.get("class")
                if label_class:
                    values.add(str(label_class))
            for prediction in item.get("predictions", []) or []:
                label_class = prediction.get("class") or prediction.get("class_name")
                if label_class:
                    values.add(str(label_class))
        for review in self.reviews.values():
            label_class = review.get("class")
            if label_class:
                values.add(str(label_class))
        return sorted(values)

    def session_meta(self) -> dict[str, Any]:
        meta_path = self.root / "session.json"
        if not meta_path.exists():
            return {}
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    def critical_indexes(self) -> list[int]:
        return [
            idx
            for idx, item in enumerate(self.manifest)
            if (item.get("critical") or {}).get("is_critical")
        ]

    def frame_payload(self, idx: int) -> dict[str, Any]:
        if not self.manifest:
            raise IndexError("empty session")
        idx = max(0, min(idx, len(self.manifest) - 1))
        item = dict(self.manifest[idx])
        frame_seq = int(item.get("frame_seq", 0))
        labels = []
        for label in item.get("labels", []) or []:
            label_item = dict(label)
            key = relabel_key(frame_seq, int(label_item.get("index", 0)))
            review = self.reviews.get(key)
            if review is not None:
                label_item["review"] = review
            label_item.setdefault("source", "model")
            labels.append(label_item)
        manual = self.manual_labels.get(str(frame_seq), [])
        manual_items = []
        for offset, entry in enumerate(manual):
            label_item = dict(entry)
            label_item["source"] = "manual"
            label_item["index"] = 10000 + offset
            label_item.setdefault("class", "objeto")
            label_item.setdefault("confidence", 1.0)
            manual_items.append(label_item)
        item["labels"] = labels + manual_items
        item["model_label_count"] = len(labels)
        item["manual_label_count"] = len(manual_items)
        item["index"] = idx
        item["count"] = len(self.manifest)
        return item

    def video_path(self) -> Path | None:
        path = (self.root / "session.mp4").resolve()
        if path.exists() and path_is_relative_to(path, self.root):
            return path
        return None

    def image_for_index(self, idx: int, *, overlay: bool) -> np.ndarray:
        item = self.frame_payload(idx)
        image = self._load_image(item)
        if image is None:
            image = placeholder_image(f"No image for frame {item.get('frame_seq')}")
        if overlay:
            image = draw_overlay(image, item)
        return image

    def _load_image(self, item: dict[str, Any]) -> np.ndarray | None:
        image_rel = item.get("image")
        if image_rel:
            image_path = (self.root / str(image_rel)).resolve()
            if image_path.exists() and path_is_relative_to(image_path, self.root):
                image = cv2.imread(str(image_path))
                if image is not None:
                    return image

        video = item.get("video") or {}
        video_rel = video.get("path")
        frame_index = video.get("frame_index")
        if video_rel is None or frame_index is None:
            return None
        video_path = (self.root / str(video_rel)).resolve()
        if not video_path.exists() or not path_is_relative_to(video_path, self.root):
            return None
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def save_session_meta(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = str(payload.get("status", "")).strip()[:60]
        notes = str(payload.get("notes", "")).strip()[:4000]
        tags_raw = payload.get("tags", [])
        if isinstance(tags_raw, str):
            tags = [part.strip() for part in tags_raw.split(",")]
        elif isinstance(tags_raw, list):
            tags = [str(item).strip() for item in tags_raw]
        else:
            tags = []
        tags = [tag[:40] for tag in tags if tag][:16]
        with self.lock:
            meta = self.session_meta()
            review = meta.get("review") if isinstance(meta.get("review"), dict) else {}
            review.update(
                {
                    "status": status or review.get("status", ""),
                    "tags": tags,
                    "notes": notes,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            meta["review"] = review
            write_json(self.root / "session.json", meta)
        return review

    def rename_frame_asset(self, idx: int, new_name: str) -> dict[str, Any]:
        if not self.manifest:
            raise IndexError("empty session")
        idx = max(0, min(idx, len(self.manifest) - 1))
        item = self.manifest[idx]
        image_rel = item.get("image")
        if not image_rel:
            raise FileNotFoundError("frame has no image asset")
        old_path = (self.root / str(image_rel)).resolve()
        if not old_path.exists() or not path_is_relative_to(old_path, self.root):
            raise FileNotFoundError("image asset not found")
        safe_name = safe_entry_name(new_name, suffix=old_path.suffix or ".jpg")
        new_path = (old_path.parent / safe_name).resolve()
        if not path_is_relative_to(new_path, self.root):
            raise ValueError("new image name escapes session")
        if new_path.exists() and new_path != old_path:
            raise FileExistsError(f"image already exists: {safe_name}")
        with self.lock:
            if new_path != old_path:
                old_path.rename(new_path)
            item["image"] = str(new_path.relative_to(self.root))
            write_jsonl(self.root / "manifest.jsonl", self.manifest)
            with (self.root / "session_edits.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "type": "rename_frame_asset",
                            "frame_index": idx,
                            "frame_seq": item.get("frame_seq"),
                            "image": item["image"],
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return {"index": idx, "frame_seq": item.get("frame_seq"), "image": item.get("image")}

    def _persist_manual_labels(self) -> None:
        path = self.root / "manual_labels.json"
        write_json(
            path,
            {
                "schema_version": 1,
                "session": str(self.root),
                "labels": self.manual_labels,
            },
        )

    def save_manual_label(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame_seq = int(payload.get("frame_seq"))
        bbox_raw = payload.get("bbox_xyxy") or []
        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            raise ValueError("bbox_xyxy must be [x1,y1,x2,y2]")
        bbox = [float(v) for v in bbox_raw]
        if bbox[2] - bbox[0] < 1.0 or bbox[3] - bbox[1] < 1.0:
            raise ValueError("bbox is too small")
        cls = str(payload.get("class", "")).strip()[:80] or "objeto"
        note = str(payload.get("note", "")).strip()[:400]
        manual_id = str(payload.get("id") or "").strip()
        track_id = payload.get("track_id")
        if track_id is not None:
            track_id = str(track_id)[:40]
        ts = datetime.now().isoformat(timespec="seconds")
        with self.lock:
            entries = self.manual_labels.setdefault(str(frame_seq), [])
            existing = next((entry for entry in entries if entry.get("id") == manual_id), None) if manual_id else None
            if existing is None:
                manual_id = manual_id or f"m{int(datetime.now().timestamp() * 1000)}-{len(entries)}"
                entry = {
                    "id": manual_id,
                    "frame_seq": frame_seq,
                    "class": cls,
                    "bbox_xyxy": [round(v, 2) for v in bbox],
                    "note": note,
                    "track_id": track_id,
                    "created_at": ts,
                    "updated_at": ts,
                    "author": "manual",
                }
                entries.append(entry)
            else:
                existing.update(
                    {
                        "class": cls,
                        "bbox_xyxy": [round(v, 2) for v in bbox],
                        "note": note,
                        "track_id": track_id,
                        "updated_at": ts,
                    }
                )
                entry = existing
            with (self.root / "manual_labels.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=True, separators=(",", ":")) + "\n")
            self._persist_manual_labels()
        return entry

    def delete_manual_label(self, frame_seq: int, manual_id: str) -> bool:
        manual_id = str(manual_id).strip()
        if not manual_id:
            raise ValueError("missing manual id")
        key = str(int(frame_seq))
        with self.lock:
            entries = self.manual_labels.get(key, [])
            kept = [item for item in entries if item.get("id") != manual_id]
            removed = len(kept) != len(entries)
            if not removed:
                return False
            if kept:
                self.manual_labels[key] = kept
            else:
                self.manual_labels.pop(key, None)
            ts = datetime.now().isoformat(timespec="seconds")
            with (self.root / "manual_labels.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"frame_seq": int(frame_seq), "id": manual_id, "deleted_at": ts},
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            self._persist_manual_labels()
        return True

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame_seq = int(payload.get("frame_seq"))
        label_index = int(payload.get("label_index"))
        key = relabel_key(frame_seq, label_index)
        review = {
            "frame_seq": frame_seq,
            "label_index": label_index,
            "class": str(payload.get("class", "")).strip(),
            "valid": bool(payload.get("valid", True)),
            "note": str(payload.get("note", "")).strip(),
        }
        with self.lock:
            self.reviews[key] = review
            with (self.root / "labels_reviewed.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(review, ensure_ascii=True, separators=(",", ":")) + "\n")
            write_json(
                self.root / "labels_reviewed.json",
                {
                    "schema_version": 1,
                    "session": str(self.root),
                    "reviews": self.reviews,
                },
            )
        return review


def placeholder_image(text: str) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :] = (10, 10, 12)
    cv2.putText(
        image,
        text[:80],
        (64, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (236, 236, 239),
        2,
        cv2.LINE_AA,
    )
    return image


def draw_overlay(image: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    output = image.copy()
    labels = item.get("labels", []) or []
    for label in labels:
        bbox = label.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        review = label.get("review") or {}
        is_valid = review.get("valid", True)
        is_manual = label.get("source") == "manual"
        if is_manual:
            color = (252, 211, 124)  # cyan-ish in BGR for manual
        else:
            color = (78, 166, 255) if is_valid else (80, 80, 240)
        label_class = review.get("class") or label.get("class") or "object"
        prefix = "M" if is_manual else str(label.get("index"))
        text = f"{prefix} #{label.get('track_id', '-')} {label_class}"
        if not is_valid and not is_manual:
            text += " reject"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            output,
            (x1, max(0, y1 - 24)),
            (min(output.shape[1] - 1, x1 + 360), y1),
            color,
            -1,
        )
        cv2.putText(
            output,
            text[:48],
            (x1 + 4, max(16, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    critical = item.get("critical") or {}
    flags = critical.get("flags") or []
    if flags:
        rules = ", ".join(str(flag.get("rule", "?")) for flag in flags[:4])
        cv2.rectangle(output, (0, 0), (output.shape[1], 34), (18, 31, 48), -1)
        cv2.putText(
            output,
            f"CRITICAL: {rules}"[:120],
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 250),
            1,
            cv2.LINE_AA,
        )
    return output


class ReplayerHandler(BaseHTTPRequestHandler):
    catalog: SessionCatalog

    server_version = "TP2SessionReplayer/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/":
            self.send_html(INDEX_HTML)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            return
        if parsed.path in {"/api/sessions", "/api/session"}:
            selected = params.get("session", [None])[0]
            sessions = [summary.__dict__ for summary in self.catalog.sessions()]
            try:
                active_id, session = self.catalog.load(selected)
                payload = {
                    "ok": True,
                    "record_root": str(self.catalog.record_root),
                    "selected_session_id": active_id,
                    "root": str(session.root),
                    "frames": len(session.manifest),
                    "critical_indexes": session.critical_indexes(),
                    "classes": session.classes(),
                    "session_meta": session.session_meta(),
                    "sessions": sessions,
                }
            except FileNotFoundError:
                payload = {
                    "ok": True,
                    "record_root": str(self.catalog.record_root),
                    "selected_session_id": None,
                    "root": None,
                    "frames": 0,
                    "critical_indexes": [],
                    "classes": [],
                    "session_meta": {},
                    "sessions": sessions,
                }
            self.send_json(payload)
            return
        if parsed.path == "/api/frame":
            selected = params.get("session", [None])[0]
            idx = int(params.get("idx", ["0"])[0])
            try:
                active_id, session = self.catalog.load(selected)
                self.send_json(
                    {
                        "ok": True,
                        "selected_session_id": active_id,
                        "frame": session.frame_payload(idx),
                    }
                )
            except (FileNotFoundError, IndexError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=404)
            return
        if parsed.path == "/frame.jpg":
            selected = params.get("session", [None])[0]
            idx = int(params.get("idx", ["0"])[0])
            overlay = params.get("overlay", ["1"])[0] != "0"
            try:
                _active_id, session = self.catalog.load(selected)
                image = session.image_for_index(idx, overlay=overlay)
            except (FileNotFoundError, IndexError):
                image = placeholder_image("No session frame available")
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if not ok:
                self.send_json({"ok": False, "error": "could not encode frame"}, status=500)
                return
            self.send_bytes(encoded.tobytes(), "image/jpeg")
            return
        if parsed.path == "/video.mp4":
            selected = params.get("session", [None])[0]
            try:
                _active_id, session = self.catalog.load(selected)
                video_path = session.video_path()
                if video_path is None:
                    raise FileNotFoundError("session.mp4 not found")
                self.send_file(video_path, "video/mp4")
            except FileNotFoundError as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=404)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/session/rename":
            try:
                payload = self.read_json_body()
                selected = str(payload.get("session_id") or "").strip()
                new_id = str(payload.get("new_id") or "").strip()
                summary = self.catalog.rename(selected, new_id)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.send_json({"ok": True, "session": summary.__dict__})
            return
        if parsed.path == "/api/session/meta":
            try:
                payload = self.read_json_body()
                selected = str(payload.get("session_id") or "").strip() or None
                active_id, session = self.catalog.load(selected)
                review = session.save_session_meta(payload)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.send_json({"ok": True, "selected_session_id": active_id, "review": review})
            return
        if parsed.path == "/api/frame/box":
            try:
                payload = self.read_json_body()
                selected = str(payload.get("session_id") or "").strip() or None
                active_id, session = self.catalog.load(selected)
                entry = session.save_manual_label(payload)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.send_json(
                {
                    "ok": True,
                    "selected_session_id": active_id,
                    "label": entry,
                    "classes": session.classes(),
                }
            )
            return
        if parsed.path == "/api/frame/box/delete":
            try:
                payload = self.read_json_body()
                selected = str(payload.get("session_id") or "").strip() or None
                frame_seq = int(payload.get("frame_seq"))
                manual_id = str(payload.get("id") or "").strip()
                active_id, session = self.catalog.load(selected)
                removed = session.delete_manual_label(frame_seq, manual_id)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.send_json({"ok": True, "selected_session_id": active_id, "removed": removed})
            return
        if parsed.path == "/api/frame/rename":
            try:
                payload = self.read_json_body()
                selected = str(payload.get("session_id") or "").strip() or None
                idx = int(payload.get("idx", 0))
                new_name = str(payload.get("new_name") or "").strip()
                active_id, session = self.catalog.load(selected)
                frame = session.rename_frame_asset(idx, new_name)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.send_json({"ok": True, "selected_session_id": active_id, "frame": frame})
            return
        if parsed.path != "/api/relabel":
            self.send_error(404, "not found")
            return
        try:
            payload = self.read_json_body()
            selected = str(payload.get("session_id") or "").strip() or None
            active_id, session = self.catalog.load(selected)
            review = session.save_review(payload)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self.send_json(
            {
                "ok": True,
                "selected_session_id": active_id,
                "review": review,
                "classes": session.classes(),
            }
        )

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(min(length, 65536)) if length > 0 else b"{}"
        payload = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("json body must be an object")
        return payload

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        self.send_bytes(body, "application/json", status=status)

    def send_html(self, html: str) -> None:
        self.send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        if size <= 0:
            self.send_bytes(b"", content_type)
            return
        start = 0
        end = size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                status = 206
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        remaining = length
        with path.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


INDEX_HTML = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TP2 / Reentrenamiento</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg-0: #08080a;
      --bg-1: #111114;
      --bg-2: #17171b;
      --bg-3: #1f1f24;
      --line: #25252c;
      --line-soft: #1b1b21;
      --line-strong: #34343d;
      --ink: #ececef;
      --ink-2: #a4a4ab;
      --ink-3: #76767f;
      --muted: #54545c;
      --blue: #4ea6ff;
      --blue-soft: rgba(78,166,255,0.16);
      --blue-deep: #1a3a78;
      --cyan: #7dd3fc;
      --teal: #5eead4;
      --amber: #fbbf24;
      --red: #f87171;
      --green: #34d399;
      --magenta: #f472b6;
      --display: "Space Grotesk", "IBM Plex Sans", -apple-system, BlinkMacSystemFont, sans-serif;
      --body: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      --shadow: 0 24px 50px rgba(0,0,0,0.55);
      --ring: 0 0 0 1px rgba(78,166,255,0.55), 0 0 0 4px rgba(78,166,255,0.10);
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      height: 100%;
      background:
        radial-gradient(1200px 700px at 88% -10%, rgba(78,166,255,0.07), transparent 60%),
        radial-gradient(900px 600px at 5% 110%, rgba(94,234,212,0.04), transparent 55%),
        var(--bg-0);
      color: var(--ink);
      font-family: var(--body);
      font-size: 13.5px;
      letter-spacing: 0;
      -webkit-font-smoothing: antialiased;
    }
    body::before {
      content: "";
      position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background-image:
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
      background-size: 32px 32px;
      mask-image: radial-gradient(circle at 50% 35%, black, transparent 75%);
    }
    .app {
      position: relative; z-index: 1;
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
      padding: 14px 18px 16px;
    }
    /* HEADER ----------------------------------------------------------------- */
    header {
      display: grid;
      grid-template-columns: minmax(280px, auto) 1fr auto;
      align-items: center;
      gap: 18px;
      padding-bottom: 11px;
      border-bottom: 1px solid var(--line);
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand .mark {
      width: 32px; height: 32px; border-radius: 8px;
      background: linear-gradient(135deg, var(--blue), var(--blue-deep));
      display: grid; place-items: center;
      box-shadow: 0 8px 22px rgba(78,166,255,0.32), inset 0 1px 0 rgba(255,255,255,0.18);
      position: relative;
    }
    .brand .mark::after {
      content: "";
      position: absolute; inset: -6px; border-radius: 14px;
      border: 1px solid rgba(78,166,255,0.18);
      pointer-events: none;
    }
    .brand .mark svg { width: 16px; height: 16px; color: #fff; }
    .brand .title { display: flex; flex-direction: column; gap: 2px; line-height: 1; }
    .brand h1 {
      margin: 0; font-family: var(--display);
      font-size: 17px; font-weight: 600; letter-spacing: -0.01em;
    }
    .brand h1 .accent { color: var(--blue); margin: 0 4px; font-weight: 400; }
    .brand h1 .sub { color: var(--ink-2); font-weight: 500; }
    .brand .meta {
      font-family: var(--mono);
      font-size: 10px; color: var(--ink-3);
      letter-spacing: 0.10em; text-transform: uppercase;
      max-width: 360px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .session-select {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 8px;
      align-items: center;
    }
    /* INPUTS / BUTTONS ------------------------------------------------------- */
    select, input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--bg-1);
      color: var(--ink);
      font-family: var(--mono);
      font-size: 12px;
      padding: 8px 10px;
      outline: none;
      transition: border-color .12s ease, box-shadow .12s ease, background .12s ease;
    }
    input[type=checkbox], input[type=radio] {
      width: auto;
      padding: 0;
      accent-color: var(--blue);
    }
    select:focus, input:focus, textarea:focus { border-color: rgba(78,166,255,0.75); box-shadow: var(--ring); }
    textarea { min-height: 60px; resize: vertical; font-family: var(--body); }
    button {
      height: 34px;
      border: 1px solid rgba(78,166,255,0.40);
      border-radius: 7px;
      background: rgba(78,166,255,0.10);
      color: var(--blue);
      font-family: var(--body);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      cursor: pointer;
      padding: 0 12px;
      display: inline-flex; align-items: center; gap: 7px;
      transition: background .12s ease, border-color .12s ease, transform .04s ease;
      white-space: nowrap;
    }
    button:hover { background: rgba(78,166,255,0.18); }
    button:active { transform: translateY(1px); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.danger { color: var(--red); border-color: rgba(248,113,113,0.5); background: rgba(248,113,113,0.10); }
    button.danger:hover { background: rgba(248,113,113,0.20); }
    button.solid { background: linear-gradient(135deg, var(--blue), #3a7fd0); color: #fff; border-color: transparent; }
    button.solid:hover { filter: brightness(1.08); }
    button.subtle { color: var(--ink-2); border-color: var(--line); background: var(--bg-2); }
    button.subtle:hover { color: var(--ink); border-color: var(--line-strong); }
    button.toggle.on { color: var(--cyan); border-color: rgba(125,211,252,0.6); background: rgba(125,211,252,0.12); }
    button .kbd { font-family: var(--mono); font-size: 9.5px; color: currentColor; opacity: .55; padding: 1px 5px; border: 1px solid currentColor; border-radius: 3px; }
    button.icon { width: 34px; padding: 0; justify-content: center; }
    button.icon svg { width: 14px; height: 14px; }
    .pills { display: flex; gap: 7px; flex-wrap: wrap; justify-content: flex-end; }
    .pill {
      display: inline-flex; align-items: center; gap: 7px;
      height: 28px; padding: 0 11px;
      border: 1px solid var(--line); border-radius: 999px;
      background: rgba(20,20,24,0.7);
      color: var(--ink-2);
      font-family: var(--mono);
      font-size: 10.5px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-weight: 600;
      white-space: nowrap;
    }
    .pill .dot { width: 6px; height: 6px; border-radius: 99px; background: currentColor; box-shadow: 0 0 10px currentColor; }
    .pill .num { color: var(--ink); font-variant-numeric: tabular-nums; }
    .pill.ok { color: var(--cyan); } .pill.warn { color: var(--amber); } .pill.bad { color: var(--red); }
    .pill.ghost { color: var(--ink-3); }
    /* MAIN GRID --------------------------------------------------------------- */
    main {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 16px;
    }
    .stage {
      min-height: 0;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 10px;
    }
    /* FRAME STAGE ------------------------------------------------------------- */
    .frame-wrap {
      min-height: 0;
      position: relative;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 14px;
      background:
        radial-gradient(120% 80% at 50% 0%, rgba(78,166,255,0.05) 0%, transparent 60%),
        radial-gradient(120% 80% at 50% 100%, rgba(0,0,0,0.4) 0%, transparent 60%),
        #0b0b0f;
      overflow: hidden;
      box-shadow: var(--shadow);
      padding: 14px;
    }
    .frame-wrap.editing {
      border-color: rgba(125,211,252,0.45);
      box-shadow: 0 0 0 1px rgba(125,211,252,0.20), var(--shadow);
    }
    .frame {
      position: relative;
      width: 100%;
      height: auto;
      max-height: 100%;
      aspect-ratio: 16 / 9;
      display: block;
      background: #050507;
      border-radius: 6px;
    }
    .frame img, .frame video {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      border-radius: 6px;
    }
    .frame.video-mode img { display: none; }
    .frame:not(.video-mode) video { display: none; }
    .frame .editor-svg {
      position: absolute;
      inset: 0;
      width: 100%; height: 100%;
      pointer-events: none;
    }
    .frame.editing .editor-svg { pointer-events: auto; cursor: crosshair; }
    .frame .corner-marks {
      position: absolute; inset: 0; border-radius: 6px; pointer-events: none;
      background:
        linear-gradient(to right, var(--blue) 0 14px, transparent 14px) top left/14px 1px no-repeat,
        linear-gradient(to bottom, var(--blue) 0 14px, transparent 14px) top left/1px 14px no-repeat,
        linear-gradient(to left, var(--blue) 0 14px, transparent 14px) top right/14px 1px no-repeat,
        linear-gradient(to bottom, var(--blue) 0 14px, transparent 14px) top right/1px 14px no-repeat,
        linear-gradient(to right, var(--blue) 0 14px, transparent 14px) bottom left/14px 1px no-repeat,
        linear-gradient(to top, var(--blue) 0 14px, transparent 14px) bottom left/1px 14px no-repeat,
        linear-gradient(to left, var(--blue) 0 14px, transparent 14px) bottom right/14px 1px no-repeat,
        linear-gradient(to top, var(--blue) 0 14px, transparent 14px) bottom right/1px 14px no-repeat;
      opacity: 0.18;
    }
    .frame-hud {
      position: absolute; left: 22px; top: 22px; z-index: 2;
      display: inline-flex; gap: 6px; align-items: center;
      padding: 6px 10px; border-radius: 999px;
      background: rgba(8,8,12,0.7); backdrop-filter: blur(8px);
      border: 1px solid var(--line);
      font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase;
      color: var(--ink-2);
      pointer-events: none;
    }
    .frame-hud .dot { width: 6px; height: 6px; border-radius: 99px; background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }
    .frame-hud.editing { color: var(--cyan); border-color: rgba(125,211,252,0.5); }
    .frame-hint {
      position: absolute; left: 50%; bottom: 26px; transform: translateX(-50%);
      padding: 7px 12px; border-radius: 999px;
      background: rgba(8,8,12,0.78); backdrop-filter: blur(8px);
      border: 1px solid rgba(125,211,252,0.4); color: var(--cyan);
      font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.06em;
      pointer-events: none;
      opacity: 0; transition: opacity .25s ease;
    }
    .frame.editing .frame-hint { opacity: 1; }
    /* DECK -------------------------------------------------------------------- */
    .deck {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(26,26,30,0.78), rgba(17,17,21,0.80));
      border-radius: 12px;
      padding: 10px 12px;
      display: grid;
      gap: 8px;
    }
    .deck .row1, .deck .row2, .deck .row3 { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .deck .row2 { gap: 12px; }
    .deck .row3 {
      gap: 10px; padding: 8px 10px;
      border: 1px dashed rgba(125,211,252,0.38);
      border-radius: 8px;
      background: rgba(125,211,252,0.04);
      color: var(--ink-2);
      font-size: 11.5px;
    }
    .deck .row3.off { display: none; }
    .deck .row3 .info b { color: var(--cyan); font-weight: 600; }
    .deck .group { display: inline-flex; align-items: center; gap: 6px; padding-right: 6px; border-right: 1px solid var(--line-soft); margin-right: 2px; }
    .deck .group:last-of-type { border-right: 0; padding-right: 0; margin-right: 0; }
    .deck .grow { flex: 1 1 auto; min-width: 0; display: flex; align-items: center; gap: 12px; }
    .deck .check { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 11px; padding: 4px 8px; border-radius: 6px; border: 1px solid transparent; cursor: pointer; }
    .deck .check:hover { border-color: var(--line); }
    .deck .speed { width: 86px; padding: 6px 8px; height: 34px; }
    .deck .pos {
      font-family: var(--mono); color: var(--ink-2); font-size: 11.5px;
      min-width: 70px; text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .deck .pos b { color: var(--ink); font-weight: 600; }
    .deck .timeline { flex: 1 1 auto; display: flex; align-items: center; gap: 10px; min-width: 200px; }
    .deck input[type=range] {
      -webkit-appearance: none; appearance: none;
      flex: 1 1 auto; height: 6px; padding: 0;
      background: linear-gradient(90deg, var(--blue) 0%, var(--blue) var(--seek-pct,0%), var(--bg-3) var(--seek-pct,0%));
      border: 1px solid var(--line);
      border-radius: 99px;
    }
    .deck input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 14px; height: 14px; border-radius: 99px;
      background: var(--ink);
      border: 2px solid var(--blue);
      box-shadow: 0 0 0 4px rgba(78,166,255,0.18);
      cursor: ew-resize;
    }
    .deck input[type=range]::-moz-range-thumb {
      width: 14px; height: 14px; border-radius: 99px;
      background: var(--ink); border: 2px solid var(--blue);
      box-shadow: 0 0 0 4px rgba(78,166,255,0.18); cursor: ew-resize;
    }
    .deck .frame-name {
      font-family: var(--mono); color: var(--ink-3); font-size: 10.5px;
      max-width: 240px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    /* SIDEBAR ---------------------------------------------------------------- */
    .side {
      min-height: 0;
      display: grid; grid-template-rows: auto auto 1fr; gap: 10px;
      padding-right: 2px;
    }
    /* CONTEXT STRIP ----------------------------------------------------------- */
    .ctx-strip {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1px;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: var(--line);
    }
    .ctx-strip .ctx {
      display: flex; flex-direction: column; gap: 2px;
      padding: 9px 12px;
      background: linear-gradient(180deg, rgba(26,26,30,0.80), rgba(17,17,21,0.80));
      min-width: 0;
    }
    .ctx-strip .ctx .label {
      font-family: var(--mono); font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase;
      color: var(--ink-3);
    }
    .ctx-strip .ctx .value {
      font-family: var(--mono); font-size: 12.5px;
      color: var(--ink); font-variant-numeric: tabular-nums;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .ctx-strip .ctx .value.tag {
      display: inline-flex; align-items: center; gap: 6px;
      font-size: 11px; font-weight: 500;
    }
    .ctx-strip .ctx .value .dot { width: 6px; height: 6px; border-radius: 99px; background: currentColor; box-shadow: 0 0 8px currentColor; }
    .ctx-strip .ctx.action-continue .value { color: var(--teal); }
    .ctx-strip .ctx.action-stop .value { color: var(--red); }
    .ctx-strip .ctx.action-turn .value { color: var(--amber); }
    .ctx-strip .ctx.action-yield .value { color: var(--cyan); }
    /* TABS ---------------------------------------------------------------- */
    .tabs {
      display: grid; grid-template-columns: repeat(3, 1fr);
      gap: 4px; padding: 4px;
      border-radius: 10px; border: 1px solid var(--line);
      background: var(--bg-2);
    }
    .tab {
      height: 30px; padding: 0 10px;
      border: 0; background: transparent;
      color: var(--ink-3);
      font-family: var(--display); font-size: 11px; font-weight: 600;
      letter-spacing: 0.14em; text-transform: uppercase;
      border-radius: 7px; cursor: pointer;
      display: flex; align-items: center; justify-content: center; gap: 7px;
    }
    .tab .badge {
      font-family: var(--mono); font-size: 9.5px; font-weight: 500;
      color: var(--ink-3);
      padding: 1px 6px; border-radius: 99px;
      background: var(--bg-1); border: 1px solid var(--line);
    }
    .tab:hover { color: var(--ink-2); }
    .tab.active { background: linear-gradient(180deg, rgba(26,26,30,0.95), rgba(17,17,21,0.95)); color: var(--ink); box-shadow: 0 0 0 1px var(--line-strong), inset 0 1px 0 rgba(255,255,255,0.04); }
    .tab.active .badge { color: var(--blue); border-color: rgba(78,166,255,0.45); }
    /* TAB PANELS ---------------------------------------------------------- */
    .tab-host {
      min-height: 0; overflow-y: auto;
      display: grid; gap: 12px; align-content: start;
      padding-right: 4px;
      scrollbar-width: thin; scrollbar-color: var(--line-strong) transparent;
    }
    .tab-host::-webkit-scrollbar { width: 8px; }
    .tab-host::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 99px; }
    .tab-panel { display: none; gap: 12px; grid-template-columns: 1fr; }
    .tab-panel.active { display: grid; animation: tabIn .25s ease both; }
    @keyframes tabIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: none; } }
    .card {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(26,26,30,0.78), rgba(17,17,21,0.78));
      border-radius: 12px;
      padding: 13px 14px 12px;
    }
    .card h2 {
      display: flex; align-items: center; gap: 8px;
      margin: 0 0 10px;
      font-family: var(--mono);
      font-weight: 600; font-size: 10.5px; letter-spacing: 0.18em; text-transform: uppercase;
      color: var(--ink-2);
    }
    .card h2 .glyph {
      width: 14px; height: 14px; display: grid; place-items: center;
      color: var(--blue);
    }
    .card h2 .badge {
      margin-left: auto;
      font-family: var(--mono);
      font-size: 10px; letter-spacing: 0.06em;
      color: var(--ink-3); padding: 2px 7px; border-radius: 999px; border: 1px solid var(--line);
      max-width: 180px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: baseline; padding: 6px 0; border-bottom: 1px solid var(--line-soft); }
    .row:last-child { border-bottom: 0; }
    .row .k { color: var(--ink-3); font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; font-size: 10.5px; }
    .row .v { font-family: var(--mono); color: var(--ink); text-align: right; font-size: 12px; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
    .flags { display: grid; gap: 6px; max-height: 240px; overflow: auto; }
    .flag {
      border: 1px solid var(--line);
      border-left: 3px solid var(--amber);
      border-radius: 7px;
      background: rgba(20,20,24,0.55);
      padding: 8px 10px;
      font-size: 12px;
      color: var(--ink-2);
    }
    .flag b { color: var(--amber); font-weight: 600; }
    .flag span { display: block; color: var(--ink-3); font-size: 10.5px; font-family: var(--mono); margin-top: 3px; word-break: break-all; }
    /* LABELS LIST --------------------------------------------------------- */
    .labels { display: grid; gap: 6px; max-height: 230px; overflow: auto; padding-right: 4px; }
    .label-row {
      display: grid; grid-template-columns: 28px 1fr auto;
      gap: 10px; align-items: center;
      border: 1px solid var(--line); border-radius: 8px;
      background: rgba(20,20,24,0.55);
      padding: 7px 10px;
      cursor: pointer;
      transition: border-color .12s, background .12s, transform .04s;
    }
    .label-row:hover { border-color: var(--line-strong); background: rgba(28,28,33,0.7); }
    .label-row .src {
      width: 26px; height: 26px; border-radius: 6px;
      background: var(--bg-3); border: 1px solid var(--line);
      display: grid; place-items: center;
      font-family: var(--mono); font-size: 10.5px; color: var(--ink-2);
      font-weight: 600;
    }
    .label-row .src.manual { color: var(--cyan); border-color: rgba(125,211,252,0.45); background: rgba(125,211,252,0.06); }
    .label-row .src.invalid { color: var(--red); border-color: rgba(248,113,113,0.45); background: rgba(248,113,113,0.06); }
    .label-row .name { color: var(--ink); font-weight: 500; line-height: 1.2; font-size: 12.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .label-row .hint { color: var(--ink-3); font-size: 10.5px; font-family: var(--mono); margin-top: 2px; }
    .label-row .key { color: var(--ink-2); font-family: var(--mono); font-size: 10.5px; }
    .label-row.active { border-color: rgba(78,166,255,0.75); background: rgba(78,166,255,0.13); }
    .label-row.active .key { color: var(--blue); }
    .label-row.invalid { border-color: rgba(248,113,113,0.45); background: rgba(248,113,113,0.10); }
    .label-row.manual { border-color: rgba(125,211,252,0.42); background: rgba(125,211,252,0.07); }
    .label-row.manual.active { border-color: rgba(125,211,252,0.85); background: rgba(125,211,252,0.16); }
    /* RELABEL FORM -------------------------------------------------------- */
    .relabel {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(20,20,24,0.7), rgba(15,15,19,0.7));
      border-radius: 12px;
      padding: 12px 14px;
      display: grid;
      gap: 10px;
    }
    .relabel .selected-bar {
      display: grid; grid-template-columns: 28px 1fr auto;
      gap: 10px; align-items: center;
      padding: 8px 10px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--bg-2);
    }
    .relabel .selected-bar .src { width: 26px; height: 26px; border-radius: 6px; background: var(--bg-3); border: 1px solid var(--line); display: grid; place-items: center; font-family: var(--mono); font-size: 10px; color: var(--ink-2); }
    .relabel .selected-bar .label { color: var(--ink); font-weight: 600; font-size: 12.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .relabel .selected-bar .meta { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); }
    .relabel .selected-bar.empty { color: var(--ink-3); font-family: var(--mono); font-size: 11px; padding: 12px; text-align: center; grid-template-columns: 1fr; }
    .class-search {
      position: relative;
    }
    .class-search input { padding-left: 30px; }
    .class-search svg {
      position: absolute; left: 9px; top: 50%; transform: translateY(-50%);
      width: 13px; height: 13px; color: var(--ink-3);
      pointer-events: none;
    }
    .quick-classes {
      display: flex; flex-wrap: wrap; gap: 5px;
      max-height: 132px; overflow-y: auto;
      padding: 2px 4px 2px 0;
      scrollbar-width: thin; scrollbar-color: var(--line-strong) transparent;
    }
    .quick-classes::-webkit-scrollbar { width: 6px; }
    .quick-classes::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 99px; }
    .chip {
      max-width: 100%;
      height: auto; min-height: 26px;
      padding: 4px 9px;
      font-family: var(--body); font-size: 10.5px; letter-spacing: 0.02em;
      text-transform: none;
      font-weight: 500;
      color: var(--ink-2);
      border: 1px solid var(--line);
      background: var(--bg-2);
      border-radius: 999px;
      cursor: pointer;
      white-space: normal;
      text-align: left;
      line-height: 1.25;
      transition: all .12s;
    }
    .chip:hover { color: var(--ink); border-color: var(--line-strong); background: var(--bg-3); }
    .chip.active { color: var(--blue); border-color: rgba(78,166,255,0.6); background: rgba(78,166,255,0.10); }
    .chip[hidden] { display: none; }
    .relabel .field-row { display: grid; grid-template-columns: 1fr; gap: 6px; }
    .relabel .toggle-line {
      display: flex; align-items: center; justify-content: space-between;
      gap: 10px;
      padding: 8px 10px;
      border: 1px solid var(--line); border-radius: 7px;
      background: var(--bg-2);
    }
    .relabel .toggle-line .lbl { font-size: 12px; color: var(--ink-2); display: inline-flex; align-items: center; gap: 8px; }
    .relabel .toggle-line .lbl .kbd { font-family: var(--mono); font-size: 9.5px; color: var(--ink-3); padding: 1px 5px; border: 1px solid var(--line); border-radius: 4px; }
    .relabel .actions { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
    .relabel .actions button { width: 100%; justify-content: center; }
    .form { display: grid; gap: 8px; }
    .inline-form { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
    .empty-state {
      padding: 14px 12px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--ink-3);
      text-align: center;
      font-size: 11.5px;
      font-family: var(--mono);
    }
    .progress {
      height: 4px; border-radius: 99px; background: var(--bg-3);
      overflow: hidden; margin-top: 8px;
    }
    .progress > div { height: 100%; background: linear-gradient(90deg, var(--blue), var(--cyan)); border-radius: 99px; transition: width .3s ease; }
    .toast {
      position: fixed; bottom: 22px; left: 50%; transform: translate(-50%, 12px);
      background: var(--bg-2); color: var(--ink); border: 1px solid var(--line);
      border-radius: 10px; padding: 9px 14px;
      font-size: 12px; letter-spacing: 0.02em;
      box-shadow: var(--shadow);
      pointer-events: none;
      opacity: 0; transition: opacity .18s ease, transform .18s ease;
      z-index: 100;
    }
    .toast.show { opacity: 1; transform: translate(-50%, 0); }
    .toast.error { border-color: rgba(248,113,113,0.55); color: var(--red); }
    .toast.ok { border-color: rgba(94,234,212,0.5); color: var(--teal); }
    /* RESPONSIVE -------------------------------------------------------------- */
    @media (max-width: 1080px) {
      .app { height: auto; }
      header { grid-template-columns: 1fr; gap: 10px; }
      .pills { justify-content: flex-start; }
      main { grid-template-columns: 1fr; }
      .stage { grid-template-rows: auto auto; }
      .frame { aspect-ratio: 16/9; max-height: 60vh; }
      .side { grid-template-rows: auto auto auto; max-height: none; overflow: visible; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="5" width="18" height="13" rx="2"/>
            <path d="M7 9l4 3-4 3M13 15h4"/>
          </svg>
        </div>
        <div class="title">
          <h1>TP2<span class="accent">/</span><span class="sub">Reentrenamiento</span></h1>
          <div class="meta" id="root">--</div>
        </div>
      </div>
      <div class="session-select">
        <select id="session-select"></select>
        <button id="refresh" class="subtle" title="Recargar catalogo">
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-3.5-7.1"/><path d="M21 4v6h-6"/></svg>
          Actualizar
        </button>
      </div>
      <div class="pills">
        <div class="pill ok" title="Frames totales"><span class="dot"></span><span class="num" id="pill-frames">0</span> fr</div>
        <div class="pill warn" title="Frames criticos"><span class="dot"></span><span class="num" id="pill-critical">0</span> crit</div>
        <div class="pill ok" title="Etiquetas revisadas"><span class="dot"></span><span class="num" id="pill-reviewed">0</span> rev</div>
        <div class="pill ghost" title="Cuadros manuales"><span class="dot"></span><span class="num" id="pill-manual">0</span> man</div>
      </div>
    </header>
    <main>
      <section class="stage">
        <div class="frame-wrap" id="frame-wrap">
          <div class="frame-hud" id="frame-hud"><span class="dot"></span><span id="hud-text">VIEW</span></div>
          <div class="frame" id="frame-shell">
            <img id="frame" alt="recorded frame">
            <video id="session-video" controls preload="metadata"></video>
            <svg class="editor-svg" id="editor-svg" preserveAspectRatio="xMidYMid meet"></svg>
            <div class="corner-marks"></div>
            <div class="frame-hint">Arrastra sobre el frame para crear un cuadro · Escape para salir</div>
          </div>
        </div>
        <div class="deck">
          <div class="row1">
            <div class="group">
              <button id="play" class="solid" title="Play / Pausa (Space)">
                <svg viewBox="0 0 24 24" width="11" height="11" fill="currentColor"><path d="M6 4l14 8-14 8z"/></svg>
                <span id="play-label">Play</span>
              </button>
              <button id="prev" class="subtle icon" title="Anterior (A / <)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="15 18 9 12 15 6"/></svg>
              </button>
              <button id="next" class="subtle icon" title="Siguiente (D / >)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="9 18 15 12 9 6"/></svg>
              </button>
              <button id="jump-back" class="subtle" title="-25 frames">-25</button>
              <button id="jump-forward" class="subtle" title="+25 frames">+25</button>
            </div>
            <div class="group">
              <button id="video-mode" class="subtle toggle" title="Reproducir session.mp4">
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
                Video
              </button>
              <button id="edit-mode" class="subtle toggle" title="Crear/borrar cuadros (E)">
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25z"/><path d="M14.06 6.19l3.75 3.75"/></svg>
                Cuadros
              </button>
            </div>
            <div class="group">
              <label class="check" title="Filtra avanzar/retroceder a frames criticos"><input id="critical" type="checkbox"> criticos</label>
              <label class="check" title="Mostrar overlay generado por el servidor"><input id="overlay" type="checkbox" checked> overlay</label>
              <select id="speed" class="speed" title="Velocidad de reproduccion">
                <option value="4">4 fps</option>
                <option value="8" selected>8 fps</option>
                <option value="12">12 fps</option>
                <option value="20">20 fps</option>
                <option value="30">30 fps</option>
              </select>
            </div>
            <div class="grow"></div>
            <span class="pos" id="pos"><b>--</b> / --</span>
          </div>
          <div class="row2">
            <div class="timeline">
              <input id="timeline" type="range" min="0" max="0" value="0">
            </div>
            <span class="frame-name" id="frame-name">--</span>
          </div>
          <div class="row3 off" id="editor-banner">
            <span class="info">Modo cuadros · arrastra sobre el frame · clase: <b id="editor-class-display">objeto</b></span>
            <span class="grow"></span>
            <button id="editor-delete" class="danger" disabled>Borrar seleccion</button>
          </div>
        </div>
      </section>
      <aside class="side">
        <div class="ctx-strip" id="ctx-strip">
          <div class="ctx" id="ctx-seq"><span class="label">Frame</span><span class="value">#--</span></div>
          <div class="ctx" id="ctx-time"><span class="label">Hora</span><span class="value">--</span></div>
          <div class="ctx" id="ctx-action"><span class="label">Accion</span><span class="value tag"><span class="dot"></span><span id="ctx-action-text">--</span></span></div>
          <div class="ctx" id="ctx-inf"><span class="label">Inferencia</span><span class="value">-- ms</span></div>
        </div>
        <nav class="tabs" role="tablist">
          <button class="tab active" data-tab="revision" role="tab">Revision <span class="badge" id="tab-badge-revision">0</span></button>
          <button class="tab" data-tab="sesion" role="tab">Sesion <span class="badge" id="tab-badge-sesion">--</span></button>
          <button class="tab" data-tab="datos" role="tab">Datos <span class="badge" id="tab-badge-datos">0</span></button>
        </nav>
        <div class="tab-host">
          <!-- ============== REVISION ============================================ -->
          <div class="tab-panel active" id="panel-revision" role="tabpanel">
            <section class="card" style="padding:12px">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><circle cx="7" cy="7" r="1.5"/></svg></span>
                Detecciones
                <span class="badge" id="label-count">0</span>
              </h2>
              <div class="labels" id="labels"><div class="empty-state">Sin labels</div></div>
            </section>
            <div class="relabel">
              <div class="selected-bar empty" id="selected-bar">Selecciona una deteccion para relabelar</div>
              <div class="class-search">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>
                <input id="class-search" placeholder="filtrar clases · escribe nueva clase">
              </div>
              <div class="quick-classes" id="quick-classes"></div>
              <div class="field-row">
                <select id="class-select"></select>
              </div>
              <div class="toggle-line">
                <span class="lbl">deteccion valida <span class="kbd">V</span></span>
                <input id="valid" type="checkbox" checked>
              </div>
              <textarea id="note" placeholder="nota opcional"></textarea>
              <div class="actions">
                <button id="save" class="solid">Guardar label · <span class="kbd">V</span></button>
                <button id="reject" class="danger">Rechazar · <span class="kbd">R</span></button>
              </div>
            </div>
          </div>
          <!-- ============== SESION ============================================== -->
          <div class="tab-panel" id="panel-sesion" role="tabpanel">
            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg></span>
                Sesion
                <span class="badge" id="session-status-badge">sin estado</span>
              </h2>
              <div class="row"><span class="k">Frames</span><span class="v" id="meta-frames">--</span></div>
              <div class="row"><span class="k">Criticos</span><span class="v" id="meta-critical">--</span></div>
              <div class="row"><span class="k">Imagenes</span><span class="v" id="meta-images">--</span></div>
              <div class="row"><span class="k">Video</span><span class="v" id="meta-video">--</span></div>
              <div class="row"><span class="k">Modificada</span><span class="v" id="meta-modified">--</span></div>
              <div class="progress" title="Progreso de revision"><div id="meta-progress" style="width:0%"></div></div>
            </section>
            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4v16h16v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></span>
                Metadatos
              </h2>
              <div class="form">
                <div class="inline-form">
                  <input id="session-name" placeholder="nombre sesion">
                  <button id="rename-session" class="subtle">Renombrar</button>
                </div>
                <select id="session-status">
                  <option value="">sin estado</option>
                  <option value="reviewing">reviewing</option>
                  <option value="ready">ready</option>
                  <option value="needs-capture">needs-capture</option>
                  <option value="discard">discard</option>
                </select>
                <input id="session-tags" placeholder="tags separados por coma">
                <textarea id="session-notes" placeholder="notas sesion"></textarea>
                <button id="save-session-meta" class="solid">Guardar sesion</button>
              </div>
            </section>
          </div>
          <!-- ============== DATOS =============================================== -->
          <div class="tab-panel" id="panel-datos" role="tabpanel">
            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 22h20L12 2z"/><path d="M12 9v5"/><circle cx="12" cy="18" r="0.6" fill="currentColor"/></svg></span>
                Flags Criticos
                <span class="badge" id="flag-count">0</span>
              </h2>
              <div class="flags" id="flags"><div class="empty-state">Sin flags</div></div>
            </section>
            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg></span>
                Frame
                <span class="badge" id="frame-badge">--</span>
              </h2>
              <div class="row"><span class="k">Seq</span><span class="v" id="frame-seq">--</span></div>
              <div class="row"><span class="k">Hora</span><span class="v" id="frame-time">--</span></div>
              <div class="row"><span class="k">Inferencia</span><span class="v" id="frame-inf">--</span></div>
              <div class="row"><span class="k">Accion</span><span class="v" id="frame-action">--</span></div>
              <div class="inline-form" style="margin-top:10px">
                <input id="asset-name" placeholder="renombrar imagen">
                <button id="rename-asset" class="subtle">Renombrar</button>
              </div>
            </section>
          </div>
        </div>
      </aside>
    </main>
  </div>
  <div class="toast" id="toast"></div>
  <script>
    const $ = id => document.getElementById(id);
    const SVG_NS = "http://www.w3.org/2000/svg";

    const state = {
      catalog: null,
      session: null,
      sessionId: null,
      idx: 0,
      selectedLabel: null,
      selectedManualId: null,
      frameData: null,
      playTimer: null,
      videoMode: false,
      editMode: false,
      drawing: null,
      classFilter: "",
      activeTab: "revision",
      imgNaturalW: 1280,
      imgNaturalH: 720,
    };

    function showToast(message, kind = "") {
      const el = $("toast");
      el.textContent = message;
      el.className = "toast show " + kind;
      clearTimeout(showToast._t);
      showToast._t = setTimeout(() => el.classList.remove("show"), 2400);
    }

    async function api(path, opts) {
      const res = await fetch(path, opts || { cache: "no-store" });
      const data = await res.json().catch(() => ({ ok: false, error: "respuesta invalida" }));
      if (!data.ok) throw new Error(data.error || "request failed");
      return data;
    }
    function enc(v) { return encodeURIComponent(v || ""); }
    function sessionQuery(extra) {
      const base = "session=" + enc(state.sessionId);
      return extra ? base + "&" + extra : base;
    }
    function currentSummary() {
      if (!state.catalog || !state.sessionId) return null;
      return (state.catalog.sessions || []).find(item => item.id === state.sessionId) || null;
    }

    /* ----------------------------- DATA / RENDER ----------------------------- */
    async function loadCatalog(preferred) {
      state.catalog = await api("/api/sessions" + (preferred ? "?session=" + enc(preferred) : ""));
      $("root").textContent = state.catalog.record_root || "--";
      const select = $("session-select");
      select.innerHTML = "";
      if (!(state.catalog.sessions || []).length) {
        const opt = document.createElement("option");
        opt.value = ""; opt.textContent = "No hay sesiones grabadas";
        select.appendChild(opt);
        renderEmpty();
        return;
      }
      for (const item of state.catalog.sessions) {
        const opt = document.createElement("option");
        opt.value = item.id;
        const status = item.review_status ? " · " + item.review_status : "";
        opt.textContent = item.id + " · " + item.frames + " fr · " + item.critical + " crit" + status;
        select.appendChild(opt);
      }
      state.sessionId = preferred || state.catalog.selected_session_id || state.catalog.sessions[0].id;
      select.value = state.sessionId;
      await loadSession(state.sessionId);
    }

    async function loadSession(id) {
      state.sessionId = id;
      state.session = await api("/api/session?session=" + enc(id));
      state.sessionId = state.session.selected_session_id;
      $("session-video").src = "/video.mp4?session=" + enc(state.sessionId) + "&t=" + Date.now();
      renderClasses(state.session.classes || []);
      renderSessionMeta();
      renderSessionEditor();
      await loadFrame(0);
    }

    function renderSessionMeta() {
      const s = currentSummary() || {};
      const frames = s.frames || state.session.frames || 0;
      const reviewed = s.reviewed || 0;
      $("pill-frames").textContent = frames;
      $("pill-critical").textContent = s.critical || (state.session.critical_indexes || []).length || 0;
      $("pill-reviewed").textContent = reviewed;
      $("meta-frames").textContent = frames || "--";
      $("meta-critical").textContent = s.critical ?? (state.session.critical_indexes || []).length;
      $("meta-images").textContent = s.images ?? "--";
      $("meta-video").textContent = s.has_video ? "session.mp4" : "--";
      $("meta-modified").textContent = s.modified_at || "--";
      const pct = frames ? Math.min(100, Math.round((reviewed / frames) * 100)) : 0;
      $("meta-progress").style.width = pct + "%";
      $("timeline").max = Math.max(0, frames - 1);
      const status = (s.review_status || "sin estado").toLowerCase();
      const badge = $("session-status-badge");
      badge.textContent = status;
      badge.style.color = status === "ready" ? "var(--teal)" : status === "discard" ? "var(--red)" : status === "reviewing" ? "var(--amber)" : "var(--ink-3)";
      $("tab-badge-sesion").textContent = status === "sin estado" ? (s.modified_at ? s.modified_at.slice(5, 10) : "--") : status.slice(0, 6);
    }

    function renderSessionEditor() {
      const meta = state.session.session_meta || {};
      const review = meta.review || {};
      $("session-name").value = state.sessionId || "";
      $("session-status").value = review.status || "";
      $("session-tags").value = (review.tags || []).join(", ");
      $("session-notes").value = review.notes || "";
    }

    function renderClasses(classes) {
      const sel = $("class-select");
      const known = classes && classes.length ? Array.from(new Set(classes)).sort() : ["objeto"];
      sel.innerHTML = "";
      const blank = document.createElement("option");
      blank.value = ""; blank.textContent = "(sin clase)";
      sel.appendChild(blank);
      for (const name of known) {
        const opt = document.createElement("option");
        opt.value = name; opt.textContent = name;
        sel.appendChild(opt);
      }
      const wrap = $("quick-classes");
      wrap.innerHTML = "";
      for (const name of known) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "chip";
        btn.dataset.cls = name.toLowerCase();
        btn.textContent = name;
        btn.onclick = () => activateClass(name);
        wrap.appendChild(btn);
      }
      filterClasses();
    }

    function activateClass(name) {
      $("class-select").value = name;
      $("class-search").value = name;
      state.classFilter = name.toLowerCase();
      filterClasses(true);
      if (state.editMode) {
        $("editor-class-display").textContent = name;
        showToast("Clase para nuevo cuadro: " + name, "ok");
      } else if (state.selectedLabel != null) {
        saveReview(true);
      }
    }

    function filterClasses(highlightExact = false) {
      const q = state.classFilter.trim();
      const wrap = $("quick-classes");
      const exact = $("class-select").value.toLowerCase();
      wrap.querySelectorAll(".chip").forEach(chip => {
        const cls = chip.dataset.cls;
        chip.hidden = q && !cls.includes(q);
        chip.classList.toggle("active", cls === exact && (highlightExact || !q || cls.includes(q)));
      });
    }

    function criticalIndexes() { return state.session ? state.session.critical_indexes || [] : []; }

    function visibleIndex(delta) {
      if (!$("critical").checked) {
        return Math.max(0, Math.min(state.session.frames - 1, state.idx + delta));
      }
      const list = criticalIndexes();
      if (!list.length) return state.idx;
      const cur = list.indexOf(state.idx);
      const next = cur < 0 ? (delta >= 0 ? 0 : list.length - 1) : Math.max(0, Math.min(list.length - 1, cur + Math.sign(delta)));
      return list[next];
    }

    function stopPlayback() {
      if (state.playTimer) clearInterval(state.playTimer);
      state.playTimer = null;
      $("play-label").textContent = "Play";
    }

    function togglePlayback() {
      if (state.playTimer) { stopPlayback(); return; }
      if (state.editMode) { showToast("Sal del modo Cuadros para reproducir", "error"); return; }
      const fps = Math.max(1, Number($("speed").value || 8));
      $("play-label").textContent = "Pausa";
      state.playTimer = setInterval(() => {
        const next = visibleIndex(1);
        if (next === state.idx || next >= state.session.frames - 1) stopPlayback();
        loadFrame(next);
      }, 1000 / fps);
    }

    function setVideoMode(on) {
      state.videoMode = !!on;
      stopPlayback();
      $("frame-shell").classList.toggle("video-mode", state.videoMode);
      $("video-mode").classList.toggle("on", state.videoMode);
      if (state.videoMode) $("session-video").play().catch(() => {});
      else $("session-video").pause();
      updateHud();
    }

    function setEditMode(on) {
      state.editMode = !!on;
      stopPlayback();
      const wrap = $("frame-wrap");
      wrap.classList.toggle("editing", state.editMode);
      $("frame-shell").classList.toggle("editing", state.editMode);
      $("edit-mode").classList.toggle("on", state.editMode);
      $("editor-banner").classList.toggle("off", !state.editMode);
      if (state.editMode) {
        if (state.videoMode) setVideoMode(false);
        $("overlay").checked = false;
      } else {
        $("overlay").checked = true;
      }
      loadFrame(state.idx);
      updateHud();
    }

    function updateHud() {
      const hud = $("frame-hud");
      const text = $("hud-text");
      hud.classList.toggle("editing", state.editMode);
      if (state.editMode) text.textContent = "EDIT";
      else if (state.videoMode) text.textContent = "VIDEO";
      else text.textContent = "VIEW";
    }

    function setTab(name) {
      state.activeTab = name;
      document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === "panel-" + name));
    }

    async function loadFrame(newIdx) {
      if (!state.sessionId || !state.session || !state.session.frames) { renderEmpty(); return; }
      state.idx = Math.max(0, Math.min(state.session.frames - 1, newIdx));
      const data = await api("/api/frame?" + sessionQuery("idx=" + state.idx));
      state.frameData = data.frame;
      state.selectedLabel = null;
      state.selectedManualId = null;
      $("timeline").value = state.idx;
      const max = Number($("timeline").max) || 1;
      $("timeline").style.setProperty("--seek-pct", ((state.idx / max) * 100).toFixed(2) + "%");
      const overlay = $("overlay").checked && !state.editMode ? "1" : "0";
      $("frame").src = "/frame.jpg?" + sessionQuery("idx=" + state.idx + "&overlay=" + overlay) + "&t=" + Date.now();
      renderFrame();
    }

    function renderFrame() {
      const f = state.frameData;
      $("pos").innerHTML = "<b>" + (state.idx + 1) + "</b> / " + state.session.frames;
      const filename = (f.image || "frame " + (f.frame_seq ?? state.idx)).split("/").pop();
      $("frame-name").textContent = filename;

      // Context strip
      $("ctx-seq").querySelector(".value").textContent = "#" + (f.frame_seq ?? "--");
      const isoTime = (f.iso_time || "").split("T")[1] || (f.iso_time || "--");
      $("ctx-time").querySelector(".value").textContent = isoTime;
      const autonomy = f.autonomy || {};
      const action = autonomy.action || (autonomy.decision || {}).action || "--";
      $("ctx-action-text").textContent = action;
      const ctxAction = $("ctx-action");
      ctxAction.className = "ctx";
      if (/stop|halt|brake/.test(action)) ctxAction.classList.add("action-stop");
      else if (/turn|left|right/.test(action)) ctxAction.classList.add("action-turn");
      else if (/yield|slow/.test(action)) ctxAction.classList.add("action-yield");
      else ctxAction.classList.add("action-continue");
      $("ctx-inf").querySelector(".value").textContent = f.inference_latency_ms == null ? "-- ms" : f.inference_latency_ms + " ms";

      // Datos tab values
      $("frame-badge").textContent = "#" + (f.frame_seq ?? "--");
      $("frame-seq").textContent = f.frame_seq ?? "--";
      $("frame-time").textContent = f.iso_time || "--";
      $("frame-inf").textContent = f.inference_latency_ms == null ? "--" : f.inference_latency_ms + " ms";
      $("frame-action").textContent = action;
      $("asset-name").value = f.image ? f.image.split("/").pop() : "";

      // Flags
      const flags = ((f.critical || {}).flags) || [];
      $("flag-count").textContent = flags.length;
      $("tab-badge-datos").textContent = flags.length;
      $("flags").innerHTML = flags.length ? "" : '<div class="empty-state">Sin flags</div>';
      for (const flag of flags) {
        const div = document.createElement("div");
        div.className = "flag";
        const detail = Object.entries(flag).filter(([k]) => k !== "rule").map(([k, v]) => k + ": " + JSON.stringify(v)).join(" · ");
        div.innerHTML = "<b>" + (flag.rule || "?") + "</b><span>" + detail + "</span>";
        $("flags").appendChild(div);
      }

      // Labels list
      const labels = f.labels || [];
      const manual = labels.filter(l => l.source === "manual").length;
      $("pill-manual").textContent = manual;
      $("label-count").textContent = labels.length + (manual ? " · " + manual + "M" : "");
      $("tab-badge-revision").textContent = labels.length;
      const list = $("labels");
      list.innerHTML = labels.length ? "" : '<div class="empty-state">Sin labels en este frame · entra en <b>Cuadros</b> para crear uno</div>';
      labels.forEach((label, i) => {
        const review = label.review || {};
        const isManual = label.source === "manual";
        const isInvalid = review.valid === false;
        const div = document.createElement("div");
        div.className = "label-row" + (isInvalid ? " invalid" : "") + (isManual ? " manual" : "") + (state.selectedLabel === i ? " active" : "");
        const cls = review.class || label.class || "objeto";
        const conf = label.confidence == null ? "--" : Math.round(label.confidence * 100) + "%";
        const trackId = label.track_id == null ? "-" : label.track_id;
        const srcMark = isManual ? "M" : isInvalid ? "X" : (i + 1);
        const srcClass = isManual ? "src manual" : isInvalid ? "src invalid" : "src";
        div.innerHTML =
          '<span class="' + srcClass + '">' + srcMark + '</span>' +
          '<span><span class="name">' + cls + '</span><div class="hint">#' + trackId + ' · ' + conf + (isManual ? " · manual" : "") + '</div></span>' +
          '<span class="key">' + (isManual ? "m" : (i + 1) + "↩") + '</span>';
        div.onclick = () => selectLabel(i);
        list.appendChild(div);
      });
      if (labels.length && state.selectedLabel == null) selectLabel(0, false);
      else updateSelectedBar();
      drawEditorOverlay();
    }

    function updateSelectedBar() {
      const bar = $("selected-bar");
      if (state.selectedLabel == null || !state.frameData) {
        bar.className = "selected-bar empty";
        bar.textContent = "Selecciona una deteccion para relabelar";
        return;
      }
      const label = state.frameData.labels[state.selectedLabel];
      if (!label) return;
      const review = label.review || {};
      const isManual = label.source === "manual";
      const cls = review.class || label.class || "objeto";
      const conf = label.confidence == null ? "--" : Math.round(label.confidence * 100) + "%";
      bar.className = "selected-bar";
      bar.innerHTML =
        '<span class="src ' + (isManual ? "manual" : "") + '" style="border-color:' + (isManual ? "rgba(125,211,252,0.5)" : "var(--line-strong)") + '">' + (isManual ? "M" : state.selectedLabel + 1) + '</span>' +
        '<span class="label">' + cls + '</span>' +
        '<span class="meta">#' + (label.track_id ?? "-") + ' · ' + conf + (isManual ? " · manual" : "") + '</span>';
    }

    function renderEmpty() {
      state.session = { frames: 0, critical_indexes: [] };
      $("frame").removeAttribute("src");
      $("pos").innerHTML = "<b>--</b> / --";
      $("frame-name").textContent = "--";
      $("timeline").value = 0;
      $("timeline").max = 0;
      $("pill-frames").textContent = "0";
      $("pill-critical").textContent = "0";
      $("pill-reviewed").textContent = "0";
      $("pill-manual").textContent = "0";
      $("labels").innerHTML = '<div class="empty-state">Sin sesiones</div>';
      $("flags").innerHTML = '<div class="empty-state">Sin flags</div>';
    }

    function selectLabel(i, rerender = true) {
      state.selectedLabel = i;
      const label = state.frameData.labels[i];
      if (!label) return;
      const review = label.review || {};
      const isManual = label.source === "manual";
      const cls = review.class || label.class || "";
      $("class-select").value = cls;
      $("class-search").value = cls;
      state.classFilter = cls.toLowerCase();
      $("valid").checked = review.valid !== false;
      $("note").value = review.note || label.note || "";
      state.selectedManualId = isManual ? label.id : null;
      $("editor-delete").disabled = !isManual;
      filterClasses(true);
      updateSelectedBar();
      if (rerender) {
        document.querySelectorAll(".label-row").forEach((row, idx) => row.classList.toggle("active", idx === i));
        drawEditorOverlay();
      }
    }

    /* ----------------------------- API SAVES --------------------------------- */
    async function saveReview(validOverride) {
      if (state.selectedLabel == null || !state.frameData) {
        showToast("Selecciona una deteccion primero", "error");
        return;
      }
      const label = state.frameData.labels[state.selectedLabel];
      if (!label || label.source === "manual") {
        showToast("Edita cuadros manuales en modo Cuadros", "error");
        return;
      }
      const cls = $("class-search").value.trim() || $("class-select").value || label.class || "";
      const valid = validOverride == null ? $("valid").checked : validOverride;
      try {
        const data = await api("/api/relabel", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: state.sessionId,
            frame_seq: state.frameData.frame_seq,
            label_index: label.index,
            class: cls,
            valid,
            note: $("note").value,
          }),
        });
        renderClasses(data.classes);
        await loadCatalog(state.sessionId);
        await loadFrame(state.idx);
        showToast(valid ? "Label guardada" : "Deteccion rechazada", "ok");
      } catch (err) { showToast(err.message, "error"); }
    }

    async function saveManualBox(bbox, cls) {
      try {
        const data = await api("/api/frame/box", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: state.sessionId,
            frame_seq: state.frameData.frame_seq,
            class: cls || "objeto",
            bbox_xyxy: bbox,
          }),
        });
        renderClasses(data.classes);
        await loadCatalog(state.sessionId);
        await loadFrame(state.idx);
        showToast("Cuadro creado · " + (cls || "objeto"), "ok");
      } catch (err) { showToast(err.message, "error"); }
    }

    async function deleteSelectedManual() {
      if (!state.selectedManualId) return;
      try {
        await api("/api/frame/box/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: state.sessionId,
            frame_seq: state.frameData.frame_seq,
            id: state.selectedManualId,
          }),
        });
        await loadCatalog(state.sessionId);
        await loadFrame(state.idx);
        showToast("Cuadro borrado", "ok");
      } catch (err) { showToast(err.message, "error"); }
    }

    async function renameSession() {
      const nextName = $("session-name").value.trim();
      if (!nextName || nextName === state.sessionId) return;
      try {
        const data = await api("/api/session/rename", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: state.sessionId, new_id: nextName }),
        });
        await loadCatalog(data.session.id);
        showToast("Sesion renombrada", "ok");
      } catch (err) { showToast(err.message, "error"); }
    }

    async function saveSessionMeta() {
      try {
        const data = await api("/api/session/meta", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: state.sessionId,
            status: $("session-status").value,
            tags: $("session-tags").value,
            notes: $("session-notes").value,
          }),
        });
        state.session.session_meta = state.session.session_meta || {};
        state.session.session_meta.review = data.review;
        await loadCatalog(state.sessionId);
        showToast("Metadatos guardados", "ok");
      } catch (err) { showToast(err.message, "error"); }
    }

    async function renameAsset() {
      if (!state.frameData) return;
      const newName = $("asset-name").value.trim();
      if (!newName) return;
      try {
        await api("/api/frame/rename", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: state.sessionId, idx: state.idx, new_name: newName }),
        });
        await loadFrame(state.idx);
        showToast("Imagen renombrada", "ok");
      } catch (err) { showToast(err.message, "error"); }
    }

    /* ----------------------------- BBOX EDITOR ------------------------------- */
    function eventToImageCoords(event) {
      const svg = $("editor-svg");
      const rect = svg.getBoundingClientRect();
      if (!rect.width || !rect.height) return [0, 0];
      const x = ((event.clientX - rect.left) / rect.width) * state.imgNaturalW;
      const y = ((event.clientY - rect.top) / rect.height) * state.imgNaturalH;
      return [Math.max(0, Math.min(state.imgNaturalW, x)), Math.max(0, Math.min(state.imgNaturalH, y))];
    }

    function drawEditorOverlay() {
      const svg = $("editor-svg");
      svg.setAttribute("viewBox", "0 0 " + state.imgNaturalW + " " + state.imgNaturalH);
      svg.innerHTML = "";
      if (!state.editMode || !state.frameData) return;
      const labels = state.frameData.labels || [];
      labels.forEach((label, i) => {
        if (!Array.isArray(label.bbox_xyxy) || label.bbox_xyxy.length !== 4) return;
        const [x1, y1, x2, y2] = label.bbox_xyxy;
        const w = x2 - x1, h = y2 - y1;
        const isManual = label.source === "manual";
        const isSelected = (isManual && state.selectedManualId === label.id) || state.selectedLabel === i;
        const stroke = isManual ? "#7dd3fc" : "#4ea6ff";
        const fill = isSelected ? "rgba(125,211,252,0.18)" : "rgba(0,0,0,0)";
        const g = document.createElementNS(SVG_NS, "g");
        g.setAttribute("data-label-i", i);
        g.style.cursor = isManual ? "pointer" : "default";

        const rect = document.createElementNS(SVG_NS, "rect");
        rect.setAttribute("x", x1);
        rect.setAttribute("y", y1);
        rect.setAttribute("width", w);
        rect.setAttribute("height", h);
        rect.setAttribute("fill", fill);
        rect.setAttribute("stroke", stroke);
        rect.setAttribute("stroke-width", isSelected ? 4 : 2.5);
        rect.setAttribute("stroke-dasharray", isManual ? "0" : "8 6");
        rect.setAttribute("vector-effect", "non-scaling-stroke");
        g.appendChild(rect);

        const tagH = Math.max(18, Math.min(34, state.imgNaturalH * 0.04));
        const tag = document.createElementNS(SVG_NS, "rect");
        tag.setAttribute("x", x1);
        tag.setAttribute("y", Math.max(0, y1 - tagH));
        tag.setAttribute("width", Math.min(state.imgNaturalW - x1, Math.max(120, (label.class || "objeto").length * 12 + 36)));
        tag.setAttribute("height", tagH);
        tag.setAttribute("fill", stroke);
        tag.setAttribute("opacity", "0.95");
        g.appendChild(tag);

        const text = document.createElementNS(SVG_NS, "text");
        text.setAttribute("x", x1 + 8);
        text.setAttribute("y", Math.max(tagH * 0.7, y1 - tagH * 0.3));
        text.setAttribute("font-family", "IBM Plex Mono, monospace");
        text.setAttribute("font-size", Math.max(11, Math.min(20, state.imgNaturalH * 0.026)));
        text.setAttribute("fill", "#0a0a0c");
        text.setAttribute("font-weight", "600");
        text.textContent = (isManual ? "M " : "") + (label.class || "objeto");
        g.appendChild(text);

        if (isManual) {
          g.addEventListener("mousedown", e => { e.stopPropagation(); selectLabel(i); });
        }
        svg.appendChild(g);
      });
    }

    function startDraw(event) {
      if (!state.editMode) return;
      if (event.button !== 0) return;
      event.preventDefault();
      const [x, y] = eventToImageCoords(event);
      state.drawing = { x0: x, y0: y, x1: x, y1: y };
      const svg = $("editor-svg");
      let rect = svg.querySelector("#draw-temp");
      if (!rect) {
        rect = document.createElementNS(SVG_NS, "rect");
        rect.setAttribute("id", "draw-temp");
        rect.setAttribute("fill", "rgba(125,211,252,0.16)");
        rect.setAttribute("stroke", "#7dd3fc");
        rect.setAttribute("stroke-width", 3);
        rect.setAttribute("stroke-dasharray", "4 4");
        rect.setAttribute("vector-effect", "non-scaling-stroke");
        svg.appendChild(rect);
      }
      rect.setAttribute("x", x);
      rect.setAttribute("y", y);
      rect.setAttribute("width", 0);
      rect.setAttribute("height", 0);
    }

    function moveDraw(event) {
      if (!state.drawing) return;
      const [x, y] = eventToImageCoords(event);
      state.drawing.x1 = x; state.drawing.y1 = y;
      const rect = $("editor-svg").querySelector("#draw-temp");
      if (!rect) return;
      const x0 = Math.min(state.drawing.x0, x);
      const y0 = Math.min(state.drawing.y0, y);
      rect.setAttribute("x", x0);
      rect.setAttribute("y", y0);
      rect.setAttribute("width", Math.abs(x - state.drawing.x0));
      rect.setAttribute("height", Math.abs(y - state.drawing.y0));
    }

    function endDraw(event) {
      if (!state.drawing) return;
      const draw = state.drawing;
      state.drawing = null;
      const svg = $("editor-svg");
      const tmp = svg.querySelector("#draw-temp");
      if (tmp) tmp.remove();
      const x1 = Math.min(draw.x0, draw.x1);
      const y1 = Math.min(draw.y0, draw.y1);
      const x2 = Math.max(draw.x0, draw.x1);
      const y2 = Math.max(draw.y0, draw.y1);
      if (x2 - x1 < state.imgNaturalW * 0.012 || y2 - y1 < state.imgNaturalH * 0.012) return;
      const cls = $("class-search").value.trim() || $("class-select").value || "objeto";
      saveManualBox([x1, y1, x2, y2], cls);
    }

    /* ----------------------------- WIRING ------------------------------------ */
    document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => setTab(t.dataset.tab)));
    $("session-select").onchange = e => loadSession(e.target.value);
    $("refresh").onclick = () => loadCatalog(state.sessionId);
    $("play").onclick = togglePlayback;
    $("prev").onclick = () => loadFrame(visibleIndex(-1));
    $("next").onclick = () => loadFrame(visibleIndex(1));
    $("jump-back").onclick = () => loadFrame(visibleIndex(-25));
    $("jump-forward").onclick = () => loadFrame(visibleIndex(25));
    $("video-mode").onclick = () => setVideoMode(!state.videoMode);
    $("edit-mode").onclick = () => setEditMode(!state.editMode);
    $("editor-delete").onclick = () => deleteSelectedManual();
    $("timeline").oninput = e => loadFrame(Number(e.target.value));
    $("overlay").onchange = () => loadFrame(state.idx);
    $("critical").onchange = () => {
      if ($("critical").checked && state.frameData && !((state.frameData.critical || {}).is_critical)) {
        loadFrame(visibleIndex(1));
      }
    };
    $("save").onclick = () => saveReview(null);
    $("reject").onclick = () => saveReview(false);
    $("rename-session").onclick = () => renameSession();
    $("save-session-meta").onclick = () => saveSessionMeta();
    $("rename-asset").onclick = () => renameAsset();
    $("class-search").addEventListener("input", e => {
      state.classFilter = e.target.value.toLowerCase();
      $("class-select").value = e.target.value.trim();
      filterClasses();
    });
    $("class-select").onchange = e => {
      $("class-search").value = e.target.value;
      state.classFilter = e.target.value.toLowerCase();
      filterClasses(true);
      if (state.editMode) $("editor-class-display").textContent = e.target.value || "objeto";
    };

    const svgEl = $("editor-svg");
    svgEl.addEventListener("mousedown", startDraw);
    svgEl.addEventListener("mousemove", moveDraw);
    svgEl.addEventListener("mouseup", endDraw);
    svgEl.addEventListener("mouseleave", endDraw);

    $("frame").addEventListener("load", () => {
      const img = $("frame");
      if (img.naturalWidth && img.naturalHeight) {
        state.imgNaturalW = img.naturalWidth;
        state.imgNaturalH = img.naturalHeight;
        $("frame-shell").style.aspectRatio = img.naturalWidth + " / " + img.naturalHeight;
        drawEditorOverlay();
      }
    });
    $("frame").addEventListener("error", () => {
      showToast("No se pudo cargar el frame", "error");
    });

    window.addEventListener("keydown", e => {
      const tag = (e.target && e.target.tagName || "").toLowerCase();
      if (["input", "textarea", "select"].includes(tag)) {
        if (e.key === "Escape" && state.editMode) setEditMode(false);
        return;
      }
      if (e.key === "ArrowLeft" || e.key.toLowerCase() === "a") loadFrame(visibleIndex(-1));
      else if (e.key === "ArrowRight" || e.key.toLowerCase() === "d") loadFrame(visibleIndex(1));
      else if (e.key === " ") { e.preventDefault(); togglePlayback(); }
      else if (e.key.toLowerCase() === "v") { $("valid").checked = true; saveReview(true); }
      else if (e.key.toLowerCase() === "r") saveReview(false);
      else if (e.key.toLowerCase() === "e") setEditMode(!state.editMode);
      else if (e.key === "Escape") { if (state.editMode) setEditMode(false); else stopPlayback(); }
      else if (e.key === "Delete" || e.key === "Backspace") {
        if (state.editMode && state.selectedManualId) deleteSelectedManual();
      } else if (e.key === "Tab") {
        e.preventDefault();
        const order = ["revision", "sesion", "datos"];
        const dir = e.shiftKey ? -1 : 1;
        const idx = order.indexOf(state.activeTab);
        setTab(order[(idx + dir + order.length) % order.length]);
      } else {
        const digit = Number(e.key);
        if (digit >= 1 && digit <= 9 && state.frameData && state.frameData.labels && state.frameData.labels[digit - 1]) {
          selectLabel(digit - 1);
        }
      }
    });

    $("timeline").addEventListener("input", e => {
      const max = Number(e.target.max) || 1;
      e.target.style.setProperty("--seek-pct", ((Number(e.target.value) / max) * 100).toFixed(2) + "%");
    });

    const params = new URLSearchParams(location.search);
    loadCatalog(params.get("session")).catch(err => showToast(err.message, "error"));
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay and relabel TP2 recorded sessions.")
    parser.add_argument(
        "record_root",
        nargs="?",
        type=Path,
        default=DEFAULT_RECORD_ROOT,
        help="Session root directory, or one specific session directory with manifest.jsonl",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    catalog = SessionCatalog(args.record_root)
    ReplayerHandler.catalog = catalog
    server = ThreadingHTTPServer((args.host, args.port), ReplayerHandler)
    print(f"Replayer listening on http://{args.host}:{args.port}/")
    print(f"Session root: {catalog.record_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
