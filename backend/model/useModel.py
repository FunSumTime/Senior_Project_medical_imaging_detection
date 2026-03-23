# functions to load a model, make the model for a heatmap and constant
from .model_loader import load_model
from .gradcam import make_gradcam_heatmap
from .config import IMG_SIZE

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageDraw
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# cache the model so we dont load all the time
_MODEL_CACHE = {}


# function to see if we need to grab the model
def get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        model_path = os.path.join(BASE_DIR, "models", model_name)
        _MODEL_CACHE[model_name] = load_model(model_path)
    return _MODEL_CACHE[model_name]

# preprocces the image to have the corners be gone only would work for lungs images i have
def mask_corners_single(img_np, frac=0.12, top_frac=0.10):
    """
    img_np: (H, W, 3) float32 or uint8
    """
    h, w, _ = img_np.shape
    fw = int(w * frac)
    th = int(h * top_frac)

    out = img_np.copy()
    out[:, :fw, :] = 0          # left strip
    out[:, w-fw:, :] = 0        # right strip
    out[:th, :, :] = 0          # top strip
    return out


# we are given a pillow image,make it into a tensor
def preprocess_pil_image(img: Image.Image, apply_mask=True):
    """
    Returns:
      image_tensor: (1, H, W, 3) float32
      vis_img: plain RGB uint8 image for drawing boxes
    """
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # convert to a numpy array
    vis_img = np.array(img).astype(np.uint8)

    model_img = vis_img.astype(np.float32)

    # Match training if you used masking
    if apply_mask:
        model_img = mask_corners_single(model_img, frac=0.12, top_frac=0.10)
    # make it into a tensor
    image_tensor = tf.convert_to_tensor(np.expand_dims(model_img, axis=0), dtype=tf.float32)
    return image_tensor, vis_img


# take a heatmap and make contuours (regoins of intrests)
def extract_boxes_from_heatmap(
    heatmap,
    image_width,
    image_height,
    threshold=0.35,
    min_area=100,
    kernel_size=3
):
    # resize the heatmap to the size of the orginal image
    hm_resized = cv2.resize(heatmap, (image_width, image_height))
    hm_resized = np.clip(hm_resized, 0.0, 1.0)
    # mask of the heatmap to get colors out
    mask = (hm_resized >= threshold).astype(np.uint8) * 255
    # kernel to 3*3 on image
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # pull out the regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image_width * image_height

 
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # if the box will be really big or small dont add them
        if area < min_area:
            continue
        if area > 0.25 * img_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": float(area)
        })

    boxes.sort(key=lambda b: b["area"], reverse=True)
    return boxes, mask


def draw_boxes_on_pil_image(pil_img, boxes, color=(255, 0, 0), width=4):
    # take the image and make a copy to draw red boxes on it.
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)

    for i, box in enumerate(boxes):
        x1 = box["x"]
        y1 = box["y"]
        x2 = x1 + box["w"]
        y2 = y1 + box["h"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1, max(0, y1 - 16)), f"ROI {i+1}", fill=color)

    return out

def make_overlay_image(vis_img, heatmap, alpha=0.35):
    hm_resized = cv2.resize(heatmap, (vis_img.shape[1], vis_img.shape[0]))
    hm_uint8 = np.uint8(255 * np.clip(hm_resized, 0, 1))
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

    # cv2 uses BGR, so convert original RGB to BGR first
    vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    overlay_bgr = cv2.addWeighted(vis_bgr, 1 - alpha, hm_color, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay_rgb, hm_resized
    # the "main" funciton
def do_cycle(model_name, image, threshold=0.5, cam_layer_name="feat_maps",Task="Xray"):
    # load the model
    model = get_model(model_name)

    # preprocess the image to get the tensor for teh model
    if Task == "Xray":
        apply_mask = True
    else:
        apply_mask = False
    image_tensor, vis_img = preprocess_pil_image(image, apply_mask=apply_mask)

    # make a prediction with the model
    preds = model.predict(image_tensor, verbose=0)[0]
    # see which one got predicted
    pred_class = int(np.argmax(preds))
    p_pneumonia = float(preds[1])
    # if its greater than our threshold count it as a anomally
    detected = p_pneumonia >= threshold

    print("raw preds:", preds)
    print("pred_class:", pred_class)
    print(f"class0: {preds[0]:.8f}")
    print(f"class1: {preds[1]:.8f}")

    # make the gradcam heatmap for the image
    heatmap = make_gradcam_heatmap(
        images=image_tensor,
        model=model,
        conv_name=cam_layer_name,
        class_index=pred_class   # match Colab behavior
    )

    print("heatmap min/max:", float(np.min(heatmap)), float(np.max(heatmap)))
    # have the heatmap on the image and get the overall heatmap for debugging
    overlay_img_np, hm_resized = make_overlay_image(vis_img, heatmap, alpha=0.35)
    overlay_img = Image.fromarray(overlay_img_np)
    # take the heatmap and get the numbers out
    boxes, mask = extract_boxes_from_heatmap(
        heatmap=heatmap,
        image_width=vis_img.shape[1],
        image_height=vis_img.shape[0],
        threshold=0.35,
        min_area=100
    )

    print("num boxes:", len(boxes))
    print("boxes:", boxes[:5])

    # keep only the biggest box for now
    if boxes:
        boxes = [boxes[0]]

    # i would like to add another image so i they can toggle between the heatmap image and the box image
    base_img = Image.fromarray(vis_img)
    boxed_img = draw_boxes_on_pil_image(base_img, boxes)

    return {
        "status": detected,
        "probability": p_pneumonia,
        "predicted_class": pred_class,
        "image_heatmap": overlay_img,
        "image_box": boxed_img,
        "boxes": boxes,
        "threshold": threshold
    }