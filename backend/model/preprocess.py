import numpy as np
import tensorflow as tf
from PIL import Image
from config import IMG_SIZE


# turn images  that come in to what  we  need 
def pil_to_float_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    arr = np.array(image).astype("float32")
    return arr

# resize the image to a numpy array
def resize_image_np(image_np: np.ndarray, target_size=IMG_SIZE) -> np.ndarray:
    tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    resized = tf.image.resize(tensor, target_size)
    return resized.numpy()


def to_batch(image_np: np.ndarray) -> tf.Tensor:
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    return tf.expand_dims(image_tensor, axis=0)