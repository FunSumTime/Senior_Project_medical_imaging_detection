import tensorflow as tf
# build a model that maps input of top conv to prediction
# grab the featuer maps and the prediction to do gradients to make a heatmap so
# it passes this into the model grad_model like a camera saying let me grab that.
def make_grad_model(model, conv_name):
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_name).output, model.output]
    )

def make_gradcam_heatmap(images, model, conv_name, class_index=None):
    with tf.GradientTape() as tape:
            # conv_out is the featuer maps
    # preds is the probailtes for the clasess [N,P]
        grad_model = make_grad_model(model, conv_name)
        conv_out, preds = grad_model(images, training=False)
              # get the prediction
    # print(preds)
    # tape will track this tensor because we wherent able to refrence it
        tape.watch(conv_out)
 # what class are we looking for
      # models confidince in th efirst image
        if class_index is None:
            class_index = tf.argmax(preds[0])

        score = preds[0, class_index]

    grads = tape.gradient(score, conv_out)
  
  # get the image

    conv_out = conv_out[0]
  # compute gradeints with featuer maps high means a lot of contribution

    grads = grads[0]
  # turn the wieghts or gradients into one weight per channel

    weights = tf.reduce_mean(grads, axis=(0, 1))
  # multiply each channel by its weight and sum over them
    cam = tf.reduce_sum(conv_out * weights, axis=-1)
  # keep the postive numbers
    cam = tf.nn.relu(cam)
  
  # normalize

    cam = cam / (tf.reduce_max(cam) + 1e-9)

    return cam.numpy()