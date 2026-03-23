import tensorflow as tf
from .config import IMG_SIZE

def center_crop(images, labels, crop=0.90):
    # images: (B,224,224,3)
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    ch = tf.cast(tf.cast(h, tf.float32) * crop, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * crop, tf.int32)
    images = tf.image.resize_with_crop_or_pad(images, ch, cw)
    images = tf.image.resize(images, (IMG_SIZE, IMG_SIZE))
    return images, labels

def mask_corners(images, labels, frac=0.12, top_frac=0.10):
    """
    frac: how much of each side (left/right) and bottom corners to mask
    top_frac: mask a top strip where many markers live (optional)
    """
    b = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    c = tf.shape(images)[3]

    mask = tf.ones((b, h, w, c), dtype=images.dtype)

    fh = tf.cast(tf.cast(h, tf.float32) * frac, tf.int32)
    fw = tf.cast(tf.cast(w, tf.float32) * frac, tf.int32)
    th = tf.cast(tf.cast(h, tf.float32) * top_frac, tf.int32)

    zeros_left  = tf.zeros((b, h, fw, c), dtype=images.dtype)
    zeros_right = tf.zeros((b, h, fw, c), dtype=images.dtype)
    zeros_top   = tf.zeros((b, th, w, c), dtype=images.dtype)

    # left strip
    mask = tf.concat([zeros_left, mask[:, :, fw:, :]], axis=2)
    # right strip
    mask = tf.concat([mask[:, :, :-fw, :], zeros_right], axis=2)
    # top strip (often contains markers)
    mask = tf.concat([zeros_top, mask[:, th:, :, :]], axis=1)

    images_masked = images * mask
    return images_masked, labels