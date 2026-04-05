# run the full piepline two model  approach
import numpy as np
import tensorflow as tf

from config import (
    GRID_ROWS,
    GRID_COLS,
    TOP_K_CELLS,
    PAD_RATIO,
    CLASS_NAMES,
    PNEUMONIA_CLASS_INDEX,
)
from .gradcam import make_gradcam_heatmap, resize_heatmap_to_image
from .preprocess import pil_to_float_image, resize_image_np, to_batch
from .image_utils import overlay_heatmap_on_image, draw_box_on_image

# do on  onne  image
def predict_single_image(model, image_np):
    image_batch = to_batch(image_np)
    preds = model(image_batch, training=False).numpy()
    return preds

# see  where the heatmap had  hot spots
def score_heatmap_grid(resized_heatmap, rows=GRID_ROWS, cols=GRID_COLS):
    h, w = resized_heatmap.shape
    cell_h = h // rows
    cell_w = w // cols

    cells = []

    for r in range(rows):
        for c in range(cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < rows - 1 else h

            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < cols - 1 else w

            cell_region = resized_heatmap[y1:y2, x1:x2]
            score = float(np.mean(cell_region))

            cells.append({
                "row": r,
                "col": c,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": score
            })

    return cells

# select the hot spots of the map
def select_top_cells(cells, top_k=TOP_K_CELLS):
    return sorted(cells, key=lambda x: x["score"], reverse=True)[:top_k]

# make a area of  region 
def merge_cells_to_box(selected_cells):
    x1 = min(cell["x1"] for cell in selected_cells)
    y1 = min(cell["y1"] for cell in selected_cells)
    x2 = max(cell["x2"] for cell in selected_cells)
    y2 = max(cell["y2"] for cell in selected_cells)
    return (x1, y1, x2, y2)


def expand_box(box, image_shape, pad_ratio=PAD_RATIO):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )

# crop to that area
def crop_and_resize(image_np, box, target_size=(224, 224)):
    x1, y1, x2, y2 = box
    crop_np = image_np[y1:y2, x1:x2]
    crop_resized = resize_image_np(crop_np, target_size=target_size)
    return crop_np, crop_resized


def run_two_stage_pipeline(
    pil_image,
    model1,
    model2,
    model1_cam_layer="feat_maps",
    model2_cam_layer="feat_maps",
    threshold=0.50,
):
    image_np = pil_to_float_image(pil_image)

    # -------------------------
    # Stage 1
    # -------------------------
    preds1 = predict_single_image(model1, image_np)
    # pulling out what the model predicted
    pred1_idx = int(np.argmax(preds1[0]))
    pred1_conf = float(preds1[0][pred1_idx])
    pneumonia_prob = float(preds1[0][PNEUMONIA_CLASS_INDEX])

    # what gets sent back to front end
    result = {
        "status": bool(pneumonia_prob >= threshold),
        "predicted_class": CLASS_NAMES[pred1_idx],
        "probability": pneumonia_prob,
        "threshold": threshold,
        "stage2_ran": False,
        "boxes": [],
        "image_heatmap": None,
        "image_box": None,
        "stage1_probs": preds1[0].tolist(),
    }

    # if stage 2 should not run, still return original-ish outputs
    if pneumonia_prob < threshold:
        result["image_heatmap"] = overlay_heatmap_on_image(
            image_np=image_np,
            heatmap_resized=np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32),
            alpha=0.0,
        )
        result["image_box"] = draw_box_on_image(image_np, None)
        return result

    # -------------------------
    # Stage 1 heatmap
    # -------------------------
    image_batch = to_batch(image_np)
    heatmap1, _ = make_gradcam_heatmap(
        model=model1,
        image_batch=image_batch,
        class_index=PNEUMONIA_CLASS_INDEX,
        layer_name=model1_cam_layer,
    )
    # now have the heatmap of the imageg blow it up
    resized_heatmap1 = resize_heatmap_to_image(heatmap1, image_np)
    # divde imagge  into  cells and score  them  with the heatmap
    cells = score_heatmap_grid(resized_heatmap1)
    selected_cells = select_top_cells(cells)
    merged_box = merge_cells_to_box(selected_cells)
    expanded_box = expand_box(merged_box, image_np.shape)

    result["boxes"] = [expanded_box]

    # stage 1 visuals
    heatmap_img = overlay_heatmap_on_image(image_np, resized_heatmap1, alpha=0.45)
    box_img = draw_box_on_image(image_np, expanded_box)

    # -------------------------
    # Stage 2 (second model)
    # -------------------------
    crop_np, crop_resized = crop_and_resize(image_np, expanded_box)
    crop_batch = to_batch(crop_resized)

    preds2 = model2(crop_batch, training=False).numpy()
    pred2_idx = int(np.argmax(preds2[0]))
    pred2_conf = float(preds2[0][pred2_idx])

    heatmap2, _ = make_gradcam_heatmap(
        model=model2,
        image_batch=crop_batch,
        class_index=PNEUMONIA_CLASS_INDEX,
        layer_name=model2_cam_layer,
    )

    heatmap2_resized = resize_heatmap_to_image(heatmap2, crop_np)

    # paste stage2 refined heatmap back into original image space
    full_refined = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
    x1, y1, x2, y2 = expanded_box
    full_refined[y1:y2, x1:x2] = heatmap2_resized

    refined_heatmap_img = overlay_heatmap_on_image(image_np, full_refined, alpha=0.50)

    result.update({
        "stage2_ran": True,
        "stage2_predicted_class": CLASS_NAMES[pred2_idx],
        "stage2_probability": float(preds2[0][PNEUMONIA_CLASS_INDEX]),
        "stage2_probs": preds2[0].tolist(),
        "image_heatmap": refined_heatmap_img,
        "image_box": box_img,
    })

    return result