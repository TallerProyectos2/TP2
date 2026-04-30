from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from session_replayer import ReplayerHandler, SessionCatalog, SessionData


def write_session(root: Path, name: str, *, critical: bool = False, image: bool = False) -> Path:
    session = root / name
    session.mkdir(parents=True)
    image_rel = None
    if image:
        images = session / "images"
        images.mkdir()
        image_rel = "images/frame_00000001.jpg"
        (session / image_rel).write_bytes(b"fake-jpg")
    item = {
        "frame_seq": 1,
        "image": image_rel,
        "labels": [
            {
                "index": 0,
                "class": "STOP",
                "confidence": 0.5,
                "bbox_xyxy": [1, 1, 10, 10],
            }
        ],
        "critical": {"is_critical": critical, "flags": [{"rule": "low_confidence_band"}] if critical else []},
    }
    (session / "manifest.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    (session / "session.json").write_text('{"started_at":"2026-04-28T10:00:00"}\n', encoding="utf-8")
    return session


class SessionCatalogTest(unittest.TestCase):
    def test_catalog_lists_sessions_from_record_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_session(root, "20260428-100000", critical=True)
            write_session(root, "20260428-101000")

            catalog = SessionCatalog(root)
            sessions = catalog.sessions()

            self.assertEqual(len(sessions), 2)
            self.assertEqual({item.id for item in sessions}, {"20260428-100000", "20260428-101000"})
            self.assertEqual(catalog.load("20260428-100000")[1].critical_indexes(), [0])

    def test_catalog_accepts_specific_session_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = write_session(Path(tmp), "20260428-100000")

            catalog = SessionCatalog(session)

            self.assertEqual(catalog.record_root, session.parent.resolve())
            self.assertEqual(catalog.latest_session_id(), "20260428-100000")

    def test_catalog_renames_session_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_session(root, "20260428-100000")

            summary = SessionCatalog(root).rename("20260428-100000", "curated-stop-left")

            self.assertEqual(summary.id, "curated-stop-left")
            self.assertTrue((root / "curated-stop-left" / "manifest.jsonl").exists())


class ReplayerHandlerTest(unittest.TestCase):
    def test_api_sessions_and_frame_use_selected_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_session(root, "20260428-100000", critical=True)
            write_session(root, "20260428-101000")
            ReplayerHandler.catalog = SessionCatalog(root)
            server = ThreadingHTTPServer(("127.0.0.1", 0), ReplayerHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                conn.request("GET", "/api/sessions?session=20260428-100000")
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["selected_session_id"], "20260428-100000")
                self.assertEqual(len(payload["sessions"]), 2)

                conn.request("GET", "/api/frame?session=20260428-100000&idx=0")
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["frame"]["frame_seq"], 1)
            finally:
                server.shutdown()
                server.server_close()


class SessionDataEditTest(unittest.TestCase):
    def test_session_meta_and_frame_asset_rename_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_path = write_session(Path(tmp), "20260428-100000", image=True)
            session = SessionData.load(session_path)

            review = session.save_session_meta(
                {"status": "ready", "tags": "stop, left", "notes": "curated"}
            )
            renamed = session.rename_frame_asset(0, "stop-left-001.jpg")

            self.assertEqual(review["status"], "ready")
            self.assertEqual(review["tags"], ["stop", "left"])
            self.assertEqual(renamed["image"], "images/stop-left-001.jpg")
            self.assertTrue((session_path / "images/stop-left-001.jpg").exists())
            manifest = json.loads((session_path / "manifest.jsonl").read_text().splitlines()[0])
            self.assertEqual(manifest["image"], "images/stop-left-001.jpg")


if __name__ == "__main__":
    unittest.main()
