import base64
import io
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from .model.useModel import do_cycle

app = Flask(__name__)
CORS(app)

# consts for the model and threshold will be able to be changed
MODEL_NAME = "TestMode.keras"
THRESHOLD = 0.50
CAM_LAYER_NAME = "feat_maps"

# simple route to see if server is up
@app.get("/api/health")
def health():
    return {"ok": True}


# main route that will take a image adn do the prediction and give out the image with all the things
@app.post("/api/predict")
def predict():
    t0 = time.perf_counter()

    if "image" not in request.files:
        return jsonify({"error": "Missing form-data file field 'image'"}), 400

    f = request.files["image"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400



    # image is good so we can do the full procces of giving it ot the model and predicting
    try:
        THRESHOLD = float(request.form.get("threshold"))
        # Task = request.form.get("taks")
        

        output = do_cycle(
            model_name=MODEL_NAME,
            image=img,
            threshold=THRESHOLD,
            cam_layer_name=CAM_LAYER_NAME
            
        )
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    result_img = output["image_heatmap"] if output["image_heatmap"] is not None else img
    w, h = result_img.size

    metrics = {
    "anomaly_detected": output["status"],
    "probability": output["probability"],
    "predicted_class": output["predicted_class"],
    "threshold": output["threshold"],
    "model": MODEL_NAME,
    "cam_layer": CAM_LAYER_NAME,
    "image_size": {"width": w, "height": h},
}

    # make a buffer for the image
    buf_heat = io.BytesIO()
    result_img.save(buf_heat, format="PNG")
    png_bytes = buf_heat.getvalue()
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    box_img = output["image_box"] if output["image_box"] is not None else img


    buf_box = io.BytesIO()
    box_img.save(buf_box, format="PNG")
    bytes_box = buf_box.getvalue()
    b64_box = base64.b64encode(bytes_box).decode("utf-8")
    

    t1 = time.perf_counter()
    # how long did this take?
    metrics["inference_ms"] = round((t1 - t0) * 1000, 2)

    # return the image and the boxes along with it.
    return jsonify({
        "result_mime": "image/png",
        "result_image_base64_heatmap": b64,
        "result_image_base64_box": b64_box,
        "metrics": metrics,
        "boxes": output["boxes"]
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)