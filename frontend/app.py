"""
frontend/app.py — Flask backend for FORENIX AI Dashboard

Endpoints
---------
GET  /                        → serve index.html
GET  /api/summary             → DB stats
GET  /api/detections          → paginated detections with filters
GET  /api/videos              → list ingested videos
POST /api/upload              → upload + analyse a video (streaming SSE progress)
GET  /api/stream/<job_id>     → SSE stream for pipeline progress
GET  /api/export/csv          → download detections as CSV
DELETE /api/videos/<video_id> → remove a video + its detections

Run
---
    cd your_project_root
    python frontend/app.py

The server looks for DB + model in the parent directory by default.
"""

from __future__ import annotations

import csv
import io
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from flask import (
    Flask, Response, jsonify, request,
    send_file, send_from_directory, stream_with_context,
)
from flask_cors import CORS

# ── Path setup: frontend/ lives one level inside project root ──────────────
ROOT = Path(__file__).parent.parent          # project root
sys.path.insert(0, str(ROOT))               # so we can import db, pipeline, etc.

from db import get_conn, init_db, summary, search, DB_PATH
from pipeline import run_pipeline

# ── Config ────────────────────────────────────────────────────────────────
DB_FILE      = str(ROOT / "forensic_logs.sqlite")
UPLOAD_DIR   = ROOT / "footage"
MODEL_PATH   = str(ROOT / "yolov8n.pt")
ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

UPLOAD_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(Path(__file__).parent), static_url_path="")
CORS(app)  # allow same-origin requests from the HTML page

# Active pipeline jobs: job_id → queue of SSE event dicts
_jobs: dict[str, queue.Queue] = {}


# ─────────────────────────────────────────────
# Helper: push SSE event into a job queue
# ─────────────────────────────────────────────

def _push(q: queue.Queue, event: str, data: dict):
    q.put({"event": event, "data": json.dumps(data)})


def _sse_format(item: dict) -> str:
    return f"event: {item['event']}\ndata: {item['data']}\n\n"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


# ── Summary ───────────────────────────────────

@app.route("/api/summary")
def api_summary():
    try:
        init_db(DB_FILE)
        conn = get_conn(DB_FILE)
        stats = summary(conn)

        # Also pull video list
        videos = conn.execute(
            "SELECT id, filename, fps, width, height, duration_s, ingested_at FROM videos ORDER BY id DESC"
        ).fetchall()
        conn.close()

        stats["videos"] = [dict(v) for v in videos]
        return jsonify({"ok": True, "data": stats})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Detections (paginated + filterable) ───────

@app.route("/api/detections")
def api_detections():
    try:
        conn = get_conn(DB_FILE)

        obj      = request.args.get("object")
        camera   = request.args.get("camera")
        min_conf = float(request.args.get("min_conf", 0.0))
        flagged  = request.args.get("flagged") == "true"
        vid_id   = request.args.get("video_id", type=int)
        limit    = min(int(request.args.get("limit", 200)), 1000)
        offset   = int(request.args.get("offset", 0))

        sql = "SELECT * FROM detections WHERE confidence >= ?"
        params: list = [min_conf]

        if obj:
            sql += " AND object = ?"
            params.append(obj.lower())
        if camera:
            sql += " AND camera_id = ?"
            params.append(camera)
        if flagged:
            sql += " AND flagged = 1"
        if vid_id is not None:
            sql += " AND video_id = ?"
            params.append(vid_id)

        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]

        rows = conn.execute(sql, params).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE confidence >= ?" +
            (" AND object = ?" if obj else "") +
            (" AND camera_id = ?" if camera else "") +
            (" AND flagged = 1" if flagged else "") +
            (" AND video_id = ?" if vid_id is not None else ""),
            params[:-2]  # strip limit/offset
        ).fetchone()[0]

        conn.close()
        return jsonify({
            "ok": True,
            "total": total,
            "offset": offset,
            "limit": limit,
            "rows": [dict(r) for r in rows],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Video list ────────────────────────────────

@app.route("/api/videos")
def api_videos():
    try:
        conn = get_conn(DB_FILE)
        videos = conn.execute(
            "SELECT v.*, COUNT(d.id) as detection_count, SUM(d.flagged) as flagged_count "
            "FROM videos v LEFT JOIN detections d ON d.video_id = v.id "
            "GROUP BY v.id ORDER BY v.id DESC"
        ).fetchall()
        conn.close()
        return jsonify({"ok": True, "videos": [dict(v) for v in videos]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Upload + analyse video ────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "No video file in request"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"ok": False, "error": f"Unsupported format: {ext}"}), 400

    # Save uploaded file
    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / safe_name
    f.save(str(save_path))

    # Pipeline settings from form
    camera_id    = request.form.get("camera_id", "CAM-01")
    confidence   = float(request.form.get("confidence", 0.50))
    sample_every = int(request.form.get("sample_every", 2))
    infer_size   = int(request.form.get("infer_size", 640))

    # Create job
    job_id = uuid.uuid4().hex
    q: queue.Queue = queue.Queue()
    _jobs[job_id] = q

    def run():
        try:
            _push(q, "status", {"stage": "init", "msg": "Initialising pipeline…", "pct": 2})

            # Monkey-patch print to capture pipeline progress
            original_print = __builtins__["print"] if isinstance(__builtins__, dict) else print
            import builtins

            pct_map = {
                "[1/6]": 5,  "[2/6]": 15, "[3/6]": 30,
                "[4/6]": 45, "[5/6]": 80, "[6/6]": 95,
            }

            def progress_print(*args, **kwargs):
                line = " ".join(str(a) for a in args)
                for tag, pct in pct_map.items():
                    if tag in line:
                        _push(q, "status", {"stage": tag, "msg": line.strip(), "pct": pct})
                        break
                else:
                    if "frame" in line and "speed" in line:
                        _push(q, "progress", {"msg": line.strip()})
                end = kwargs.get("end", "\n")
                if end == "\n":
                    pass  # suppress to avoid flooding

            builtins.print = progress_print

            try:
                stats = run_pipeline(
                    str(save_path),
                    camera_id=camera_id,
                    model=MODEL_PATH,
                    confidence=confidence,
                    sample_every_n=sample_every,
                    infer_size=infer_size,
                    batch_size=8,
                    db_path=DB_FILE,
                    verbose=True,
                )
            finally:
                builtins.print = original_print

            _push(q, "done", {"stats": stats, "pct": 100})

        except Exception as e:
            _push(q, "error", {"msg": str(e)})
        finally:
            # Cleanup job after 5 min
            def cleanup():
                time.sleep(300)
                _jobs.pop(job_id, None)
            threading.Thread(target=cleanup, daemon=True).start()

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


# ── SSE stream for job progress ───────────────

@app.route("/api/stream/<job_id>")
def api_stream(job_id: str):
    q = _jobs.get(job_id)
    if q is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404

    def generate():
        yield _sse_format({"event": "connected", "data": json.dumps({"job_id": job_id})})
        while True:
            try:
                item = q.get(timeout=30)
                yield _sse_format(item)
                if item["event"] in ("done", "error"):
                    break
            except queue.Empty:
                yield ": keepalive\n\n"  # SSE comment to keep connection alive

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Export CSV ────────────────────────────────

@app.route("/api/export/csv")
def api_export_csv():
    try:
        conn = get_conn(DB_FILE)
        obj      = request.args.get("object")
        flagged  = request.args.get("flagged") == "true"
        vid_id   = request.args.get("video_id", type=int)
        min_conf = float(request.args.get("min_conf", 0.0))

        rows = search(conn, object_class=obj, flagged_only=flagged,
                      video_id=vid_id, min_confidence=min_conf, limit=50000)
        conn.close()

        si = io.StringIO()
        if rows:
            writer = csv.DictWriter(si, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])

        output = io.BytesIO(si.getvalue().encode())
        output.seek(0)
        return send_file(output, mimetype="text/csv",
                         as_attachment=True,
                         download_name="forenix_detections.csv")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Delete video + detections ─────────────────

@app.route("/api/videos/<int:video_id>", methods=["DELETE"])
def api_delete_video(video_id: int):
    try:
        conn = get_conn(DB_FILE)
        conn.execute("DELETE FROM detections WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Camera list ───────────────────────────────

@app.route("/api/cameras")
def api_cameras():
    try:
        conn = get_conn(DB_FILE)
        cams = conn.execute(
            "SELECT DISTINCT camera_id, COUNT(*) as cnt FROM detections GROUP BY camera_id"
        ).fetchall()
        conn.close()
        return jsonify({"ok": True, "cameras": [dict(c) for c in cams]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ─────────────────────────────────────────────
if __name__ == "__main__":
    init_db(DB_FILE)
    print(f"\n  FORENIX AI — Dashboard server")
    print(f"  DB   : {DB_FILE}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Open : http://localhost:5000\n")
    app.run(debug=True, port=5000, threaded=True)