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
        return cls(root=root, manifest=manifest, reviews=reviews, lock=threading.RLock())

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
        labels = []
        for label in item.get("labels", []) or []:
            label_item = dict(label)
            key = relabel_key(int(item.get("frame_seq", 0)), int(label_item.get("index", 0)))
            review = self.reviews.get(key)
            if review is not None:
                label_item["review"] = review
            labels.append(label_item)
        item["labels"] = labels
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
        color = (78, 166, 255) if is_valid else (80, 80, 240)
        label_class = review.get("class") or label.get("class") or "object"
        text = f"{label.get('index')} #{label.get('track_id', '-')} {label_class}"
        if not is_valid:
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
  <title>TP2 · Reentrenamiento</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg-0: #0a0a0c;
      --bg-1: #131316;
      --bg-2: #1a1a1e;
      --line: #26262b;
      --line-soft: #1c1c20;
      --ink: #ececef;
      --ink-2: #a4a4ab;
      --muted: #61616a;
      --blue: #4ea6ff;
      --blue-deep: #1a3a78;
      --cyan: #7dd3fc;
      --amber: #fbbf24;
      --red: #f87171;
      --body: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      --shadow: 0 24px 50px rgba(0,0,0,0.55);
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      min-height: 100%;
      background:
        radial-gradient(1200px 700px at 92% -10%, rgba(78,166,255,0.06), transparent 60%),
        var(--bg-0);
      color: var(--ink);
      font-family: var(--body);
      font-size: 13.5px;
      letter-spacing: 0;
      -webkit-font-smoothing: antialiased;
    }
    .app { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; gap: 18px; padding: 20px 22px 22px; }
    header {
      display: grid;
      grid-template-columns: minmax(260px, auto) 1fr auto;
      align-items: center;
      gap: 22px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }
    .brand { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .brand .mark {
      width: 36px; height: 36px; border-radius: 8px;
      background: linear-gradient(135deg, var(--blue), var(--blue-deep));
      display: grid; place-items: center;
      box-shadow: 0 6px 18px rgba(78,166,255,0.28), inset 0 1px 0 rgba(255,255,255,0.12);
    }
    .brand .mark svg { width: 18px; height: 18px; color: #fff; }
    .brand h1 { margin: 0; font-size: 19px; line-height: 1.1; font-weight: 600; letter-spacing: -0.005em; }
    .brand h1 .accent { color: var(--blue); margin: 0 4px; font-weight: 400; }
    .brand h1 .sub { color: var(--ink-2); font-weight: 500; }
    .brand .meta { font-family: var(--mono); font-size: 10.5px; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }
    .session-select {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 10px;
      align-items: center;
    }
    select, input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--bg-1);
      color: var(--ink);
      font-family: var(--mono);
      font-size: 12px;
      padding: 9px 10px;
      outline: none;
    }
    textarea { min-height: 70px; resize: vertical; font-family: var(--body); }
    button {
      height: 38px;
      border: 1px solid rgba(78,166,255,0.42);
      border-radius: 7px;
      background: rgba(78,166,255,0.10);
      color: var(--blue);
      font-family: var(--body);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      cursor: pointer;
      padding: 0 14px;
    }
    button:hover { background: rgba(78,166,255,0.16); }
    button.danger { color: var(--red); border-color: rgba(248,113,113,0.5); background: rgba(248,113,113,0.12); }
    .pills { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .pill {
      display: inline-flex; align-items: center; gap: 8px;
      height: 30px; padding: 0 12px;
      border: 1px solid var(--line); border-radius: 999px;
      background: rgba(26,26,30,0.7);
      color: var(--ink-2);
      font-size: 10.5px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 600;
      white-space: nowrap;
    }
    .pill .dot { width: 7px; height: 7px; border-radius: 99px; background: currentColor; box-shadow: 0 0 12px currentColor; }
    .pill.ok { color: var(--cyan); } .pill.warn { color: var(--amber); } .pill.bad { color: var(--red); }
    main { min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 390px; gap: 20px; }
    .stage { min-height: 0; display: grid; grid-template-rows: minmax(320px, 1fr) auto; gap: 14px; }
    .frame {
      position: relative;
      min-height: 320px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: radial-gradient(120% 80% at 50% 50%, #16161a 0%, #08080a 100%);
      overflow: hidden;
      box-shadow: var(--shadow);
    }
    .frame img, .frame video { display: block; width: 100%; height: 100%; max-height: calc(100vh - 220px); object-fit: contain; }
    .frame video { background: #050507; }
    .frame.video-mode img { display: none; }
    .frame:not(.video-mode) video { display: none; }
    .frame::after {
      content: "";
      position: absolute; inset: 14px; border-radius: 8px; pointer-events: none;
      background:
        linear-gradient(to right, var(--blue) 0 14px, transparent 14px) top left/14px 1px no-repeat,
        linear-gradient(to bottom, var(--blue) 0 14px, transparent 14px) top left/1px 14px no-repeat,
        linear-gradient(to left, var(--blue) 0 14px, transparent 14px) top right/14px 1px no-repeat,
        linear-gradient(to bottom, var(--blue) 0 14px, transparent 14px) top right/1px 14px no-repeat,
        linear-gradient(to right, var(--blue) 0 14px, transparent 14px) bottom left/14px 1px no-repeat,
        linear-gradient(to top, var(--blue) 0 14px, transparent 14px) bottom left/1px 14px no-repeat,
        linear-gradient(to left, var(--blue) 0 14px, transparent 14px) bottom right/14px 1px no-repeat,
        linear-gradient(to top, var(--blue) 0 14px, transparent 14px) bottom right/1px 14px no-repeat;
      opacity: 0.22;
    }
    .deck, .card {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(26,26,30,0.7), rgba(19,19,22,0.75));
      border-radius: 12px;
    }
    .deck { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; padding: 12px; }
    .deck .left, .deck .right { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .timeline { grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 86px; gap: 12px; align-items: center; }
    .timeline input { padding: 0; accent-color: var(--blue); }
    .speed { width: 84px; padding: 8px 9px; }
    label { display: inline-flex; gap: 7px; align-items: center; color: var(--ink-2); font-size: 12px; }
    .pos { font-family: var(--mono); color: var(--ink-2); font-size: 12px; min-width: 86px; text-align: right; }
    .side { min-height: 0; overflow-y: auto; display: grid; align-content: start; gap: 14px; padding-right: 4px; }
    .card { padding: 16px 16px 14px; }
    .card h2 { margin: 0 0 12px; font-weight: 600; font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; }
    .row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: baseline; padding: 7px 0; border-bottom: 1px solid var(--line-soft); }
    .row:last-child { border-bottom: 0; }
    .row .k { color: var(--muted); font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; font-size: 10.5px; }
    .row .v { font-family: var(--mono); color: var(--ink); text-align: right; font-size: 12px; overflow-wrap: anywhere; }
    .flags, .labels { display: grid; gap: 7px; max-height: 260px; overflow: auto; }
    .flag, .label-row {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(20,20,24,0.5);
      padding: 8px 10px;
      font-size: 12px;
    }
    .flag { border-left: 3px solid var(--amber); color: var(--ink-2); }
    .label-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; cursor: pointer; }
    .label-row.active { border-color: rgba(78,166,255,0.75); background: rgba(78,166,255,0.14); }
    .label-row.invalid { border-color: rgba(248,113,113,0.55); background: rgba(248,113,113,0.12); }
    .label-row .name { color: var(--ink); font-weight: 500; }
    .label-row small { color: var(--muted); font-family: var(--mono); }
    .form { display: grid; gap: 8px; }
    .inline-form { display: grid; grid-template-columns: 1fr auto; gap: 8px; margin-top: 10px; }
    .quick-classes { display: flex; flex-wrap: wrap; gap: 7px; }
    .quick-classes button { height: 30px; padding: 0 9px; font-size: 10px; letter-spacing: 0.06em; }
    .status-select { min-width: 120px; }
    .empty { color: var(--muted); border-style: dashed; text-align: center; }
    @media (max-width: 1080px) {
      header { grid-template-columns: 1fr; gap: 14px; }
      .pills { justify-content: flex-start; }
      main { grid-template-columns: 1fr; }
      .frame img { max-height: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 6h16M4 12h16M4 18h10"/>
          </svg>
        </div>
        <div>
          <h1>TP2<span class="accent">/</span><span class="sub">Reentrenamiento</span></h1>
          <div class="meta" id="root">--</div>
        </div>
      </div>
      <div class="session-select">
        <select id="session-select"></select>
        <button id="refresh">Actualizar</button>
      </div>
      <div class="pills">
        <div class="pill ok"><span class="dot"></span><span id="pill-frames">0 fr</span></div>
        <div class="pill warn"><span class="dot"></span><span id="pill-critical">0 crit</span></div>
        <div class="pill ok"><span class="dot"></span><span id="pill-reviewed">0 rev</span></div>
      </div>
    </header>
    <main>
      <section class="stage">
        <div class="frame" id="frame-shell">
          <img id="frame" alt="recorded frame">
          <video id="session-video" controls preload="metadata"></video>
        </div>
        <div class="deck">
          <div class="left">
            <button id="play">Play</button>
            <button id="prev">Anterior</button>
            <button id="next">Siguiente</button>
            <button id="jump-back">-25</button>
            <button id="jump-forward">+25</button>
            <button id="video-mode">Video</button>
            <label><input id="critical" type="checkbox"> solo criticos</label>
            <label><input id="overlay" type="checkbox" checked> overlay</label>
            <select id="speed" class="speed">
              <option value="4">4 fps</option>
              <option value="8" selected>8 fps</option>
              <option value="12">12 fps</option>
              <option value="20">20 fps</option>
            </select>
          </div>
          <div class="right"><span class="pos" id="pos">--</span></div>
          <div class="timeline">
            <input id="timeline" type="range" min="0" max="0" value="0">
            <span class="pos" id="frame-name">--</span>
          </div>
        </div>
      </section>
      <aside class="side">
        <section class="card">
          <h2>Sesion</h2>
          <div class="row"><span class="k">Frames</span><span class="v" id="meta-frames">--</span></div>
          <div class="row"><span class="k">Criticos</span><span class="v" id="meta-critical">--</span></div>
          <div class="row"><span class="k">Imagenes</span><span class="v" id="meta-images">--</span></div>
          <div class="row"><span class="k">Video</span><span class="v" id="meta-video">--</span></div>
          <div class="row"><span class="k">Modificada</span><span class="v" id="meta-modified">--</span></div>
          <div class="inline-form">
            <input id="session-name" placeholder="nombre sesion">
            <button id="rename-session">Renombrar</button>
          </div>
          <div class="form" style="margin-top:10px">
            <select id="session-status" class="status-select">
              <option value="">sin estado</option>
              <option value="reviewing">reviewing</option>
              <option value="ready">ready</option>
              <option value="needs-capture">needs-capture</option>
              <option value="discard">discard</option>
            </select>
            <input id="session-tags" placeholder="tags separados por coma">
            <textarea id="session-notes" placeholder="notas sesion"></textarea>
            <button id="save-session-meta">Guardar sesion</button>
          </div>
        </section>
        <section class="card">
          <h2>Frame</h2>
          <div class="row"><span class="k">Seq</span><span class="v" id="frame-seq">--</span></div>
          <div class="row"><span class="k">Hora</span><span class="v" id="frame-time">--</span></div>
          <div class="row"><span class="k">Inferencia</span><span class="v" id="frame-inf">--</span></div>
          <div class="row"><span class="k">Accion</span><span class="v" id="frame-action">--</span></div>
          <div class="inline-form">
            <input id="asset-name" placeholder="renombrar imagen">
            <button id="rename-asset">Renombrar</button>
          </div>
        </section>
        <section class="card">
          <h2>Flags Criticos</h2>
          <div class="flags" id="flags"><div class="flag empty">Sin flags</div></div>
        </section>
        <section class="card">
          <h2>Labels</h2>
          <div class="labels" id="labels"><div class="label-row empty">Sin labels</div></div>
        </section>
        <section class="card">
          <h2>Relabel</h2>
          <div class="form">
            <div class="quick-classes" id="quick-classes"></div>
            <select id="class-select"></select>
            <input id="class-input" placeholder="nueva clase">
            <label><input id="valid" type="checkbox" checked> deteccion valida</label>
            <textarea id="note" placeholder="nota"></textarea>
            <button id="save">Guardar label</button>
            <button class="danger" id="reject">Rechazar deteccion</button>
          </div>
        </section>
      </aside>
    </main>
  </div>
  <script>
    const $ = id => document.getElementById(id);
    let catalog = null, session = null, sessionId = null, idx = 0, selected = null, frameData = null;
    let playTimer = null, videoMode = false;
    async function api(path, opts) {
      const res = await fetch(path, opts || {cache: 'no-store'});
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'request failed');
      return data;
    }
    function enc(v) { return encodeURIComponent(v || ''); }
    function sessionQuery(extra) { return 'session=' + enc(sessionId) + (extra ? '&' + extra : ''); }
    function currentSummary() {
      if (!catalog || !sessionId) return null;
      return (catalog.sessions || []).find(item => item.id === sessionId) || null;
    }
    async function loadCatalog(preferred) {
      catalog = await api('/api/sessions' + (preferred ? '?session=' + enc(preferred) : ''));
      $('root').textContent = catalog.record_root || '--';
      const select = $('session-select');
      select.innerHTML = '';
      if (!(catalog.sessions || []).length) {
        const opt = document.createElement('option');
        opt.value = ''; opt.textContent = 'No hay sesiones grabadas';
        select.appendChild(opt);
        renderEmpty();
        return;
      }
      for (const item of catalog.sessions) {
        const opt = document.createElement('option');
        opt.value = item.id;
        opt.textContent = item.id + ' · ' + item.frames + ' fr · ' + item.critical + ' crit';
        select.appendChild(opt);
      }
      sessionId = preferred || catalog.selected_session_id || catalog.sessions[0].id;
      select.value = sessionId;
      await loadSession(sessionId);
    }
    async function loadSession(id) {
      sessionId = id;
      session = await api('/api/session?session=' + enc(id));
      sessionId = session.selected_session_id;
      $('session-video').src = '/video.mp4?session=' + enc(sessionId) + '&t=' + Date.now();
      renderClasses(session.classes || []);
      renderSessionMeta();
      renderSessionEditor();
      await loadFrame(0);
    }
    function renderSessionMeta() {
      const s = currentSummary() || {};
      $('pill-frames').textContent = (s.frames || session.frames || 0) + ' fr';
      $('pill-critical').textContent = (s.critical || (session.critical_indexes || []).length || 0) + ' crit';
      $('pill-reviewed').textContent = (s.reviewed || 0) + ' rev';
      $('meta-frames').textContent = s.frames ?? session.frames ?? '--';
      $('meta-critical').textContent = s.critical ?? (session.critical_indexes || []).length;
      $('meta-images').textContent = s.images ?? '--';
      $('meta-video').textContent = s.has_video ? 'session.mp4' : '--';
      $('meta-modified').textContent = s.modified_at || '--';
      $('timeline').max = Math.max(0, (session.frames || 1) - 1);
    }
    function renderSessionEditor() {
      const meta = session.session_meta || {};
      const review = meta.review || {};
      $('session-name').value = sessionId || '';
      $('session-status').value = review.status || '';
      $('session-tags').value = (review.tags || []).join(', ');
      $('session-notes').value = review.notes || '';
    }
    function renderClasses(classes) {
      const sel = $('class-select');
      sel.innerHTML = '';
      $('quick-classes').innerHTML = '';
      for (const name of classes) {
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name;
        sel.appendChild(opt);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = name;
        btn.onclick = () => { $('class-select').value = name; $('class-input').value = ''; saveReview(true); };
        $('quick-classes').appendChild(btn);
      }
    }
    function criticalIndexes() { return session ? session.critical_indexes || [] : []; }
    function visibleIndex(delta) {
      if (!$('critical').checked) return Math.max(0, Math.min(session.frames - 1, idx + delta));
      const list = criticalIndexes();
      if (!list.length) return idx;
      const cur = list.indexOf(idx);
      const next = cur < 0 ? (delta >= 0 ? 0 : list.length - 1) : Math.max(0, Math.min(list.length - 1, cur + delta));
      return list[next];
    }
    function stopPlayback() {
      if (playTimer) clearInterval(playTimer);
      playTimer = null;
      $('play').textContent = 'Play';
    }
    function togglePlayback() {
      if (playTimer) { stopPlayback(); return; }
      const fps = Math.max(1, Number($('speed').value || 8));
      $('play').textContent = 'Pausa';
      playTimer = setInterval(() => {
        const next = visibleIndex(1);
        if (next === idx || next >= session.frames - 1) stopPlayback();
        loadFrame(next);
      }, 1000 / fps);
    }
    function setVideoMode(on) {
      videoMode = !!on;
      stopPlayback();
      $('frame-shell').classList.toggle('video-mode', videoMode);
      $('video-mode').textContent = videoMode ? 'Frames' : 'Video';
      if (videoMode) $('session-video').play().catch(() => {});
      else $('session-video').pause();
    }
    async function loadFrame(newIdx) {
      if (!sessionId || !session || !session.frames) { renderEmpty(); return; }
      idx = Math.max(0, Math.min(session.frames - 1, newIdx));
      const data = await api('/api/frame?' + sessionQuery('idx=' + idx));
      frameData = data.frame;
      selected = null;
      $('timeline').value = idx;
      $('frame').src = '/frame.jpg?' + sessionQuery('idx=' + idx + '&overlay=' + ($('overlay').checked ? '1' : '0')) + '&t=' + Date.now();
      renderFrame();
    }
    function renderFrame() {
      $('pos').textContent = (idx + 1) + ' / ' + session.frames;
      $('frame-name').textContent = (frameData.image || 'frame ' + (frameData.frame_seq ?? idx)).split('/').pop();
      $('frame-seq').textContent = frameData.frame_seq ?? '--';
      $('frame-time').textContent = frameData.iso_time || '--';
      $('frame-inf').textContent = frameData.inference_latency_ms == null ? '--' : frameData.inference_latency_ms + ' ms';
      const autonomy = frameData.autonomy || {};
      $('frame-action').textContent = autonomy.action || (autonomy.decision || {}).action || '--';
      $('asset-name').value = frameData.image ? frameData.image.split('/').pop() : '';
      const flags = (((frameData.critical || {}).flags) || []);
      $('flags').innerHTML = flags.length ? '' : '<div class="flag empty">Sin flags</div>';
      for (const flag of flags) {
        const div = document.createElement('div');
        div.className = 'flag';
        div.innerHTML = '<b>' + (flag.rule || '?') + '</b><br><span>' + JSON.stringify(flag) + '</span>';
        $('flags').appendChild(div);
      }
      const labels = frameData.labels || [];
      $('labels').innerHTML = labels.length ? '' : '<div class="label-row empty">Sin labels</div>';
      labels.forEach((label, i) => {
        const review = label.review || {};
        const div = document.createElement('div');
        div.className = 'label-row' + (review.valid === false ? ' invalid' : '') + (selected === i ? ' active' : '');
        const conf = label.confidence == null ? '--' : Math.round(label.confidence * 100) + '%';
        div.innerHTML = '<span><span class="name">#' + (label.track_id ?? '-') + ' ' + (review.class || label.class || 'objeto') + '</span><br><small>' + conf + '</small></span><span>' + label.index + '</span>';
        div.onclick = () => selectLabel(i);
        $('labels').appendChild(div);
      });
      if (labels.length && selected == null) selectLabel(0, false);
    }
    function renderEmpty() {
      session = {frames: 0, critical_indexes: []};
      $('frame').removeAttribute('src');
      $('pos').textContent = '--';
      $('frame-name').textContent = '--';
      $('timeline').value = 0;
      $('timeline').max = 0;
      $('pill-frames').textContent = '0 fr';
      $('pill-critical').textContent = '0 crit';
      $('pill-reviewed').textContent = '0 rev';
      $('labels').innerHTML = '<div class="label-row empty">Sin sesiones</div>';
      $('flags').innerHTML = '<div class="flag empty">Sin flags</div>';
    }
    function selectLabel(i, rerender=true) {
      selected = i;
      const label = frameData.labels[i];
      const review = label.review || {};
      $('class-select').value = review.class || label.class || '';
      $('class-input').value = '';
      $('valid').checked = review.valid !== false;
      $('note').value = review.note || '';
      if (rerender) renderFrame();
    }
    async function saveReview(validOverride) {
      if (selected == null || !frameData) return;
      const label = frameData.labels[selected];
      const cls = $('class-input').value.trim() || $('class-select').value || label.class || '';
      const valid = validOverride == null ? $('valid').checked : validOverride;
      const data = await api('/api/relabel', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId, frame_seq: frameData.frame_seq, label_index: label.index, class: cls, valid, note: $('note').value})
      });
      renderClasses(data.classes);
      await loadCatalog(sessionId);
      await loadFrame(idx);
    }
    async function renameSession() {
      const nextName = $('session-name').value.trim();
      if (!nextName || nextName === sessionId) return;
      const data = await api('/api/session/rename', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId, new_id: nextName})
      });
      await loadCatalog(data.session.id);
    }
    async function saveSessionMeta() {
      const data = await api('/api/session/meta', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_id: sessionId,
          status: $('session-status').value,
          tags: $('session-tags').value,
          notes: $('session-notes').value
        })
      });
      session.session_meta = session.session_meta || {};
      session.session_meta.review = data.review;
      await loadCatalog(sessionId);
    }
    async function renameAsset() {
      if (!frameData) return;
      const newName = $('asset-name').value.trim();
      if (!newName) return;
      await api('/api/frame/rename', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId, idx, new_name: newName})
      });
      await loadFrame(idx);
    }
    $('session-select').onchange = e => loadSession(e.target.value);
    $('refresh').onclick = () => loadCatalog(sessionId);
    $('play').onclick = togglePlayback;
    $('prev').onclick = () => loadFrame(visibleIndex(-1));
    $('next').onclick = () => loadFrame(visibleIndex(1));
    $('jump-back').onclick = () => loadFrame(visibleIndex(-25));
    $('jump-forward').onclick = () => loadFrame(visibleIndex(25));
    $('video-mode').onclick = () => setVideoMode(!videoMode);
    $('timeline').oninput = e => loadFrame(Number(e.target.value));
    $('overlay').onchange = () => loadFrame(idx);
    $('critical').onchange = () => { if ($('critical').checked && frameData && !((frameData.critical || {}).is_critical)) loadFrame(visibleIndex(1)); };
    $('save').onclick = () => saveReview(null);
    $('reject').onclick = () => saveReview(false);
    $('rename-session').onclick = () => renameSession().catch(err => alert(err.message));
    $('save-session-meta').onclick = () => saveSessionMeta().catch(err => alert(err.message));
    $('rename-asset').onclick = () => renameAsset().catch(err => alert(err.message));
    window.addEventListener('keydown', e => {
      const tag = (e.target && e.target.tagName || '').toLowerCase();
      if (['input','textarea','select'].includes(tag)) return;
      if (e.key === 'ArrowLeft' || e.key.toLowerCase() === 'a') loadFrame(visibleIndex(-1));
      if (e.key === 'ArrowRight' || e.key.toLowerCase() === 'd') loadFrame(visibleIndex(1));
      if (e.key === ' ') { e.preventDefault(); togglePlayback(); }
      if (e.key.toLowerCase() === 'v') { $('valid').checked = true; saveReview(true); }
      if (e.key.toLowerCase() === 'r') saveReview(false);
      const digit = Number(e.key);
      if (digit >= 1 && digit <= 9 && frameData && frameData.labels && frameData.labels[digit - 1]) selectLabel(digit - 1);
    });
    const params = new URLSearchParams(location.search);
    loadCatalog(params.get('session')).catch(err => alert(err.message));
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
