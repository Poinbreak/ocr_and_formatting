"""
Flask application — serves the OCR dashboard and API endpoints.
"""

import os
import sys
import json
import uuid
import traceback
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    Response,
)
from PIL import Image

# Ensure UTF-8 on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Import registry lazily to avoid heavy torch imports on module load
from model_registry import manager, MODEL_CATALOGUE, DEFAULT_SCHEMA, enhance_image

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


def _save_result(content: str, ext: str) -> str:
    """Save content to a result file and return its id."""
    rid = uuid.uuid4().hex[:12]
    fname = f"{rid}.{ext}"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return rid


# ===================================================================
# Routes — Pages
# ===================================================================

@app.route("/")
def index():
    return render_template("index.html")


# ===================================================================
# Routes — API
# ===================================================================

@app.route("/api/models", methods=["GET"])
def api_models():
    """Return the model catalogue."""
    models = []
    for key, meta in MODEL_CATALOGUE.items():
        models.append(
            {
                "key": key,
                "name": meta["name"],
                "group": meta["group"],
                "description": meta["description"],
                "default_prompt": meta["default_prompt"],
                "backend": meta["backend"],
                "available": meta.get("available", True),
            }
        )
    return jsonify(
        {
            "models": models,
            "loaded": manager.loaded_key,
            "default_schema": DEFAULT_SCHEMA,
        }
    )


@app.route("/api/ollama/status", methods=["GET"])
def api_ollama_status():
    """Check if Ollama is reachable."""
    try:
        import requests as req

        r = req.get("http://localhost:11434", timeout=2)
        return jsonify({"online": True})
    except Exception:
        return jsonify({"online": False})


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    """Run OCR on an uploaded image with chosen model."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type: {file.filename}"}), 400

    model_key = request.form.get("model", "qwen2.5-vl-3b")
    prompt = request.form.get("prompt", "")
    do_enhance = request.form.get("enhance", "false").lower() == "true"

    if model_key not in MODEL_CATALOGUE:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    if not prompt:
        prompt = MODEL_CATALOGUE[model_key]["default_prompt"]

    # Save upload
    ext = os.path.splitext(file.filename)[1]
    uid = uuid.uuid4().hex[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    file.save(save_path)

    try:
        img = Image.open(save_path).convert("RGB")

        if do_enhance:
            img = enhance_image(img)

        raw_text = manager.run_ocr(model_key, img, prompt)
        rid = _save_result(raw_text, "txt")

        return jsonify(
            {
                "text": raw_text,
                "result_id": rid,
                "model": MODEL_CATALOGUE[model_key]["name"],
                "char_count": len(raw_text),
                "word_count": len(raw_text.split()),
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        try:
            os.remove(save_path)
        except OSError:
            pass


@app.route("/api/extract", methods=["POST"])
def api_extract():
    """Run NuExtract on raw text with a JSON schema."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    schema = data.get("schema", DEFAULT_SCHEMA).strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        json.loads(schema)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON schema: {e}"}), 400

    try:
        result = manager.run_nuextract(text, schema)
        # Try to pretty-print the result
        try:
            parsed = json.loads(result.strip())
            result = json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        rid = _save_result(result, "json")
        return jsonify({"json": result, "result_id": rid})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline", methods=["POST"])
def api_pipeline():
    """Full pipeline: image → enhance → OCR → NuExtract → JSON."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type: {file.filename}"}), 400

    model_key = request.form.get("model", "qwen2.5-vl-3b")
    prompt = request.form.get("prompt", "")
    schema = request.form.get("schema", DEFAULT_SCHEMA)
    do_enhance = request.form.get("enhance", "true").lower() == "true"

    if model_key not in MODEL_CATALOGUE:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    if not prompt:
        prompt = MODEL_CATALOGUE[model_key]["default_prompt"]

    ext = os.path.splitext(file.filename)[1]
    uid = uuid.uuid4().hex[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    file.save(save_path)

    try:
        img = Image.open(save_path).convert("RGB")

        # Step 1: Enhancement
        enhanced_img = enhance_image(img) if do_enhance else img

        # Step 2: OCR
        raw_text = manager.run_ocr(model_key, enhanced_img, prompt)
        txt_id = _save_result(raw_text, "txt")

        # Step 3: NuExtract
        try:
            json.loads(schema)
        except json.JSONDecodeError:
            schema = DEFAULT_SCHEMA

        json_result = manager.run_nuextract(raw_text, schema)
        try:
            parsed = json.loads(json_result.strip())
            json_result = json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        json_id = _save_result(json_result, "json")

        return jsonify(
            {
                "text": raw_text,
                "json": json_result,
                "text_id": txt_id,
                "json_id": json_id,
                "model": MODEL_CATALOGUE[model_key]["name"],
                "enhanced": do_enhance,
                "char_count": len(raw_text),
                "word_count": len(raw_text.split()),
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(save_path)
        except OSError:
            pass


@app.route("/api/download/<file_type>/<result_id>", methods=["GET"])
def api_download(file_type, result_id):
    """Download a result file."""
    if file_type not in ("txt", "json"):
        return jsonify({"error": "Invalid file type."}), 400

    fname = f"{result_id}.{file_type}"
    path = os.path.join(RESULTS_DIR, fname)

    if not os.path.exists(path):
        return jsonify({"error": "File not found."}), 404

    return send_file(
        path,
        as_attachment=True,
        download_name=f"ocr_result.{file_type}",
    )


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  OCR & Extraction Dashboard")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
