import tensorflow as tf
from model_loader import load_model

model_cam = load_model("/models/model_cam.keras")

# build a model that maps input of top conv to prediction
# grab the featuer maps and the prediction to do gradients to make a heatmap so
# it passes this into the model grad_model like a camera saying let me grab that.
grad_model = tf.keras.Model(
    inputs = model_cam.inputs,
    outputs = [model_cam.get_layer("feat_maps").output, model_cam.output]
)

print("Grad model Ok")
# print(eff.output.shape)
# grad_model.summary()

# make the gradcam heatmap

def make_gradcam_heatmap(images,class_index=None):
  """
    images: a batch tensor shaped (B, 224, 224, 3) from dataset (already resized).
    class_index: which class to explain (0=normal, 1=pneumonia). If None, uses model's predicted class.
    Returns: heatmap for the first image in the batch, in range 0..1
    """
  with tf.GradientTape() as tape:
    # conv_out is the featuer maps
    # preds is the probailtes for the clasess [N,P]
    conv_out, preds = grad_model(images, training=False)
      # get the prediction
    # print(preds)
    # tape will track this tensor because we wherent able to refrence it
    tape.watch(conv_out)

    if class_index is None:
      class_index = tf.argmax(preds[0])
    # what class are we looking for
      # models confidince in th efirst image
    score = preds[0,class_index]

  # compute gradeints with featuer maps high means a lot of contribution
  grads = tape.gradient(score,conv_out)

  # get the image
  conv_out = conv_out[0]
  grads = grads[0]

  # turn the wieghts or gradients into one weight per channel
  weights = tf.reduce_mean(grads,axis=(0,1))

  # multiply each channel by its weight and sum over them
  cam = tf.reduce_sum(tf.multiply(conv_out,weights),axis=-1)
  # keep the postive numbers
  cam = tf.nn.relu(cam)

  # normalize
  cam = cam / (tf.reduce_max(cam) + 1e-9)
  return cam.numpy()