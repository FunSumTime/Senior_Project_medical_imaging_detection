import base64
import io
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
CORS(app)  # allow frontend (localhost) to call backend

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/predict")
def predict():
    t0 = time.perf_counter()

    if "image" not in request.files:
        return jsonify({"error": "Missing form-data file field 'image'"}), 400

    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Load image
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    # ---- TODO: replace this block with your real model inference ----
    # For now: draw a rectangle + label as a stand-in for "bbox/gradcam output"
    out = img.copy()
    draw = ImageDraw.Draw(out)

    w, h = out.size
    # A demo box in the middle
    x1, y1 = int(w * 0.2), int(h * 0.2)
    x2, y2 = int(w * 0.8), int(h * 0.8)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=max(3, w // 200))

    label = "anomaly: YES (demo)"
    # font is optional; Pillow might not have system fonts available
    draw.text((x1 + 10, y1 + 10), label, fill=(255, 60, 60))

    # fake metrics
    metrics = {
        "anomaly_detected": True,
        "probability": 0.91,
        "threshold": 0.50,
        "model": "demo-overlay",
        "image_size": {"width": w, "height": h},
    }
    # ---------------------------------------------------------------

    # Encode output to base64 PNG
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    t1 = time.perf_counter()
    metrics["inference_ms"] = round((t1 - t0) * 1000, 2)

    return jsonify({
        "result_mime": "image/png",
        "result_image_base64": b64,
        "metrics": metrics
    })

if __name__ == "__main__":
    # For dev only. In production use gunicorn/uwsgi.
    app.run(host="127.0.0.1", port=5000, debug=True)