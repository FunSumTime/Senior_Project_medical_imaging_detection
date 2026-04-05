import numpy as np
import tensorflow as tf

# gradcam is  what grabs the featur maps and then displays  what  the model learnded
def make_gradcam_heatmap(model, image_batch, class_index=None, layer_name="feat_maps"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    ) 
    # what layer  to look at

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        # if there  is  not  a given onne to look at, get what the model predicted
        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_channel = predictions[:, class_index]
    # get the gradient of  the featuer  maps
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # get  the values and then  make a heatmao
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    # ratio it to a  certainn range
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), predictions.numpy()

# take numbers from  gradcam and blow it  up
def resize_heatmap_to_image(heatmap, image_np):
    h, w = image_np.shape[:2]
    # take the heatmap make it a tensor
    heatmap_tf = tf.convert_to_tensor(heatmap, dtype=tf.float32)
    heatmap_tf = tf.expand_dims(heatmap_tf, axis=-1)
    heatmap_tf = tf.expand_dims(heatmap_tf, axis=0)

    resized = tf.image.resize(heatmap_tf, size=(h, w), method="bilinear")
    resized = tf.squeeze(resized).numpy()

    resized = np.maximum(resized, 0)
    max_val = resized.max()
    if max_val > 0:
        resized = resized / max_val

    return resized