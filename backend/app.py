import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from config import (
    DEFAULT_STAGE1_MODEL,
    DEFAULT_STAGE2_MODEL,
    DEFAULT_THRESHOLD,
    IMG_SIZE
)
from model.model_loader import load_model_by_key, get_cam_layer
from model.pipeline import run_two_stage_pipeline
from model.image_utils import pil_to_base64_png

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://senior-project-medical-imaging-dete.vercel.app", # Fixed the // and used the name from your logs
            "http://localhost:5173",  # For local site testing
            "http://localhost:3000"   # For local React testing
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/predict")
def predict():
    t0 = time.perf_counter()
    # main route to get image and send it to pipeline

    if "image" not in request.files:
        return jsonify({"error": "Missing form-data file field 'image'"}), 400

    f = request.files["image"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Open the image and convert to RGB
        img = Image.open(f.stream).convert("RGB")
        
        # ADD THIS LINE: Resize it to match what DenseNet expects
        img = img.resize(IMG_SIZE) 
        
    except Exception as e:
        return jsonify({"error": f"Could not read or resize image: {e}"}), 400

    stage1_key = request.form.get("stage1_model", DEFAULT_STAGE1_MODEL)
    stage2_key = request.form.get("stage2_model", DEFAULT_STAGE2_MODEL)

    try:
        threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))
    except ValueError:
        return jsonify({"error": "Invalid threshold"}), 400

    try:
        model1 = load_model_by_key(stage1_key)
        model2 = load_model_by_key(stage2_key)

        output = run_two_stage_pipeline(
            pil_image=img,
            model1=model1,
            model2=model2,
            model1_cam_layer=get_cam_layer(stage1_key),
            model2_cam_layer=get_cam_layer(stage2_key),
            threshold=threshold,
        )
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    heatmap_img = output["image_heatmap"] if output["image_heatmap"] is not None else img
    box_img = output["image_box"] if output["image_box"] is not None else img

    heatmap_b64 = pil_to_base64_png(heatmap_img)
    box_b64 = pil_to_base64_png(box_img)

    t1 = time.perf_counter()
    detected_label = "normal"
    if output["status"]:
        detected_label = output.get("stage2_predicted_class") or output["predicted_class"]

    metrics = {
        "anomaly_detected": output["status"],
        "probability": output["probability"],
        "predicted_class": output["predicted_class"],
        "detected_label": detected_label,
        "threshold": output["threshold"],
        "model": stage1_key,
        "stage2_model": stage2_key,
        "cam_layer": get_cam_layer(stage1_key),
        "stage2_ran": output["stage2_ran"],
        "stage2_predicted_class": output.get("stage2_predicted_class"),
        "stage2_probability": output.get("stage2_probability"),
        "inference_ms": round((t1 - t0) * 1000, 2),
    }

    return jsonify({
        "result_mime": "image/png",
        "result_image_base64_heatmap": heatmap_b64,
        "result_image_base64_box": box_b64,
        "metrics": metrics,
        "boxes": output["boxes"],
    })


import os

if __name__ == "__main__":
    # Render provides a 'PORT' environment variable. 
    # If it's not there (like on your laptop), it defaults to 5000.
    port = int(os.environ.get("PORT", 5000))
    
    # 0.0.0.0 is required for Render to "see" your app.
    app.run(host="0.0.0.0", port=port)