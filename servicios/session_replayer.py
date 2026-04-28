from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


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
            if image_path.exists() and self.root in image_path.parents:
                image = cv2.imread(str(image_path))
                if image is not None:
                    return image

        video = item.get("video") or {}
        video_rel = video.get("path")
        frame_index = video.get("frame_index")
        if video_rel is None or frame_index is None:
            return None
        video_path = (self.root / str(video_rel)).resolve()
        if not video_path.exists() or self.root not in video_path.parents:
            return None
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

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
    image[:, :] = (16, 16, 18)
    cv2.putText(
        image,
        text[:80],
        (64, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (230, 230, 235),
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
        color = (80, 220, 120) if is_valid else (80, 80, 240)
        label_class = review.get("class") or label.get("class") or "object"
        text = f"{label.get('index')} #{label.get('track_id', '-')} {label_class}"
        if not is_valid:
            text += " reject"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(output, (x1, max(0, y1 - 24)), (min(output.shape[1] - 1, x1 + 360), y1), color, -1)
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
        cv2.rectangle(output, (0, 0), (output.shape[1], 34), (30, 45, 70), -1)
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
    session: SessionData

    server_version = "TP2SessionReplayer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(INDEX_HTML)
            return
        if parsed.path == "/api/session":
            self.send_json(
                {
                    "ok": True,
                    "root": str(self.session.root),
                    "frames": len(self.session.manifest),
                    "critical_indexes": self.session.critical_indexes(),
                    "classes": self.session.classes(),
                }
            )
            return
        if parsed.path == "/api/frame":
            params = parse_qs(parsed.query)
            idx = int(params.get("idx", ["0"])[0])
            try:
                self.send_json({"ok": True, "frame": self.session.frame_payload(idx)})
            except IndexError as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=404)
            return
        if parsed.path == "/frame.jpg":
            params = parse_qs(parsed.query)
            idx = int(params.get("idx", ["0"])[0])
            overlay = params.get("overlay", ["1"])[0] != "0"
            image = self.session.image_for_index(idx, overlay=overlay)
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if not ok:
                self.send_json({"ok": False, "error": "could not encode frame"}, status=500)
                return
            self.send_bytes(encoded.tobytes(), "image/jpeg")
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/relabel":
            self.send_error(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(min(length, 65536)) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            review = self.session.save_review(payload)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self.send_json({"ok": True, "review": review, "classes": self.session.classes()})

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


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TP2 Session Replayer</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
    body { margin: 0; background: #101114; color: #f0f0f2; }
    header { display: flex; justify-content: space-between; gap: 18px; align-items: center; padding: 14px 18px; border-bottom: 1px solid #2b2d34; }
    h1 { font-size: 17px; margin: 0; font-weight: 650; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 390px; gap: 16px; padding: 16px; }
    .viewer { display: grid; gap: 12px; min-width: 0; }
    .frame { background: #08090b; border: 1px solid #2b2d34; border-radius: 8px; overflow: hidden; min-height: 320px; }
    .frame img { display: block; width: 100%; max-height: calc(100vh - 150px); object-fit: contain; }
    .bar, .panel { border: 1px solid #2b2d34; border-radius: 8px; background: #17191e; }
    .bar { display: flex; align-items: center; gap: 10px; padding: 10px; flex-wrap: wrap; }
    button, input, select, textarea { font: inherit; }
    button { border: 1px solid #3b82f6; background: #19345f; color: #f8fbff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }
    button.secondary { border-color: #444852; background: #22252c; }
    button.danger { border-color: #ef4444; background: #4a1d25; }
    input, select, textarea { background: #101114; color: #f0f0f2; border: 1px solid #3a3d46; border-radius: 6px; padding: 7px 8px; }
    label { display: inline-flex; align-items: center; gap: 6px; color: #c4c7d0; }
    aside { display: grid; align-content: start; gap: 12px; min-width: 0; }
    .panel { padding: 12px; }
    .panel h2 { margin: 0 0 10px; font-size: 12px; color: #c4c7d0; text-transform: uppercase; letter-spacing: .08em; }
    .meta { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #c4c7d0; font-size: 12px; display: grid; gap: 4px; }
    .labels { display: grid; gap: 7px; max-height: 320px; overflow: auto; }
    .label-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; border: 1px solid #30333b; border-radius: 6px; padding: 8px; cursor: pointer; }
    .label-row.active { border-color: #3b82f6; background: #172b4c; }
    .label-row.invalid { border-color: #ef4444; background: #351b22; }
    .flags { display: grid; gap: 6px; }
    .flag { border: 1px solid #4b5563; border-left: 3px solid #f59e0b; border-radius: 5px; padding: 7px; font-size: 12px; color: #e5e7eb; }
    .form { display: grid; gap: 8px; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } .frame img { max-height: none; } }
  </style>
</head>
<body>
  <header>
    <h1>TP2 Session Replayer</h1>
    <div class="meta"><span id="root">--</span></div>
  </header>
  <main>
    <section class="viewer">
      <div class="frame"><img id="frame" alt="recorded frame"></div>
      <div class="bar">
        <button id="prev">Prev</button>
        <button id="next">Next</button>
        <label><input id="critical" type="checkbox"> critical only</label>
        <label><input id="overlay" type="checkbox" checked> overlay</label>
        <span class="meta" id="pos">--</span>
      </div>
    </section>
    <aside>
      <section class="panel">
        <h2>Frame</h2>
        <div class="meta" id="frame-meta">--</div>
      </section>
      <section class="panel">
        <h2>Critical Flags</h2>
        <div class="flags" id="flags"></div>
      </section>
      <section class="panel">
        <h2>Labels</h2>
        <div class="labels" id="labels"></div>
      </section>
      <section class="panel">
        <h2>Relabel</h2>
        <div class="form">
          <select id="class-select"></select>
          <input id="class-input" placeholder="or new class">
          <label><input id="valid" type="checkbox" checked> valid detection</label>
          <textarea id="note" rows="3" placeholder="note"></textarea>
          <button id="save">Save reviewed label</button>
          <button class="danger" id="reject">Reject detection</button>
        </div>
      </section>
    </aside>
  </main>
  <script>
    let session = null, idx = 0, selected = null, frameData = null;
    const $ = id => document.getElementById(id);
    async function api(path, opts) {
      const res = await fetch(path, opts || {cache: 'no-store'});
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'request failed');
      return data;
    }
    function criticalIndexes() { return session ? session.critical_indexes : []; }
    function visibleIndex(delta) {
      if (!$('critical').checked) return Math.max(0, Math.min(session.frames - 1, idx + delta));
      const list = criticalIndexes();
      if (!list.length) return idx;
      const cur = list.indexOf(idx);
      const next = cur < 0 ? (delta >= 0 ? 0 : list.length - 1) : Math.max(0, Math.min(list.length - 1, cur + delta));
      return list[next];
    }
    async function loadSession() {
      session = await api('/api/session');
      $('root').textContent = session.root;
      renderClasses(session.classes);
      await loadFrame(0);
    }
    function renderClasses(classes) {
      const sel = $('class-select');
      sel.innerHTML = '';
      for (const name of classes) {
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name;
        sel.appendChild(opt);
      }
    }
    async function loadFrame(newIdx) {
      idx = Math.max(0, Math.min(session.frames - 1, newIdx));
      const data = await api('/api/frame?idx=' + idx);
      frameData = data.frame;
      selected = null;
      $('frame').src = '/frame.jpg?idx=' + idx + '&overlay=' + ($('overlay').checked ? '1' : '0') + '&t=' + Date.now();
      renderFrame();
    }
    function renderFrame() {
      $('pos').textContent = (idx + 1) + ' / ' + session.frames;
      $('frame-meta').innerHTML = [
        'frame_seq: ' + frameData.frame_seq,
        'time: ' + (frameData.iso_time || '--'),
        'image: ' + (frameData.image || '--'),
        'video_frame: ' + ((frameData.video && frameData.video.frame_index) ?? '--'),
        'inference_ms: ' + (frameData.inference_latency_ms ?? '--'),
        'action: ' + (((frameData.autonomy || {}).action) || ((frameData.autonomy || {}).decision || {}).action || '--')
      ].map(x => '<span>' + x + '</span>').join('');
      const flags = (((frameData.critical || {}).flags) || []);
      $('flags').innerHTML = flags.length ? flags.map(f => '<div class="flag"><b>' + (f.rule || '?') + '</b><br><span>' + JSON.stringify(f) + '</span></div>').join('') : '<div class="meta">none</div>';
      const labels = frameData.labels || [];
      $('labels').innerHTML = labels.length ? '' : '<div class="meta">no labels</div>';
      labels.forEach((label, i) => {
        const review = label.review || {};
        const div = document.createElement('div');
        div.className = 'label-row' + (review.valid === false ? ' invalid' : '') + (selected === i ? ' active' : '');
        div.innerHTML = '<span>#' + (label.track_id ?? '-') + ' ' + (review.class || label.class || 'object') + '<br><small>' + (label.confidence == null ? '--' : Math.round(label.confidence * 100) + '%') + '</small></span><span>' + label.index + '</span>';
        div.onclick = () => selectLabel(i);
        $('labels').appendChild(div);
      });
      if (labels.length) selectLabel(0, false);
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
      if (selected == null) return;
      const label = frameData.labels[selected];
      const cls = $('class-input').value.trim() || $('class-select').value || label.class || '';
      const valid = validOverride == null ? $('valid').checked : validOverride;
      const data = await api('/api/relabel', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({frame_seq: frameData.frame_seq, label_index: label.index, class: cls, valid, note: $('note').value})
      });
      renderClasses(data.classes);
      await loadFrame(idx);
    }
    $('prev').onclick = () => loadFrame(visibleIndex(-1));
    $('next').onclick = () => loadFrame(visibleIndex(1));
    $('overlay').onchange = () => loadFrame(idx);
    $('critical').onchange = () => { if ($('critical').checked && !((frameData.critical || {}).is_critical)) loadFrame(visibleIndex(1)); };
    $('save').onclick = () => saveReview(null);
    $('reject').onclick = () => saveReview(false);
    window.addEventListener('keydown', e => {
      if (e.key === 'ArrowLeft') loadFrame(visibleIndex(-1));
      if (e.key === 'ArrowRight') loadFrame(visibleIndex(1));
    });
    loadSession().catch(err => alert(err.message));
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay and relabel a TP2 recorded session.")
    parser.add_argument("session_dir", type=Path, help="Directory containing manifest.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    session = SessionData.load(args.session_dir)
    if not session.manifest:
        print(f"No frames found in {session.root / 'manifest.jsonl'}", file=sys.stderr)
        return 2

    ReplayerHandler.session = session
    server = ThreadingHTTPServer((args.host, args.port), ReplayerHandler)
    print(f"Replayer listening on http://{args.host}:{args.port}/")
    print(f"Session: {session.root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
