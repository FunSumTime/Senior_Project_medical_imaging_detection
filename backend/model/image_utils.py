import base64
import io
import numpy as np
from PIL import Image, ImageDraw


# channge from  numpy  to pillow 
def np_to_pil(image_np: np.ndarray) -> Image.Image:
    image_np = np.clip(image_np, 0, 255).astype("uint8")
    return Image.fromarray(image_np)

# swith to bas64  imgae
def pil_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# put heatmap of gradcam on image
def overlay_heatmap_on_image(image_np: np.ndarray, heatmap_resized: np.ndarray, alpha=0.40):
    """
    Simple red overlay, no OpenCV needed.
    """
    image_np = np.clip(image_np, 0, 255).astype("uint8")
    heat = np.clip(heatmap_resized, 0, 1)

    overlay = image_np.copy().astype("float32")
    red = np.zeros_like(overlay)
    red[..., 0] = 255.0

    heat_3 = np.stack([heat, heat, heat], axis=-1)

    blended = overlay * (1.0 - alpha * heat_3) + red * (alpha * heat_3)
    blended = np.clip(blended, 0, 255).astype("uint8")

    return Image.fromarray(blended)

# draw  area of  region
def draw_box_on_image(image_np: np.ndarray, box, color="red", width=3):
    pil_img = np_to_pil(image_np)
    draw = ImageDraw.Draw(pil_img)

    if box is not None:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    return pil_img