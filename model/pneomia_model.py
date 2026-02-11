import tensorflow as tf

def make_modelA(img_size=(256,256),use_augmentation=True):
    # what the model will recive 256X256 ,3 saying my model will recive these tensors

    inputs = tf.keras.Input(shape=img_size + (3,))

    # start pipeline from inputs
    x = inputs

    # (recommended) normalize to 0..1
    # rescaling because they like numbers in scale of 0-1
    x = tf.keras.layers.Rescaling(1./255)(x)

    # augmentation (training only automatically)
    if use_augmentation:
        x = data_augmentation(x)
    # 32 filters size 3*3
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    # reduce size and keep strongest inputs
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="last_conv_layer")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # the classify into two groups
    outputs = tf.keras.layers.Dense(1, activation="sigmoid",name="predictions")(x)

    model = tf.keras.Model(inputs, outputs)
    # for layer in model.layers:
    #     print(layer.name)
    #     print(layer)
    # We use the names we defined earlier in the model
    # creating a wraper around the last layer to grab its output and the model output of what it got
    # we also give the first input of what the model got
  
    return model

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
], name="augmentation")




import numpy as np



def make_grad_model(model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    return grad_model

def grad_engine(img_array, grad_model):
    with tf.GradientTape() as tape:
        # grab the outputs from the sub model, as the gradient wants to do the gradent with respect to the map based of the prediction
        last_conv_layer_output, predictions = grad_model(img_array)
        # want to grab the value of the prediction, so we go from row to row and grab the value [[0.9],[1]]...
        class_value = predictions[:,0]
    # take the gradient of the prediction with respect to the featuer maps
    # for this specific pixel how does it change the overall prediction
    grads = tape.gradient(class_value, last_conv_layer_output)
    # Shape: (1, height, width, channels) (e.g., (1, 16, 16, 256))

    # calculate the mean(average) of the pixels to get the weight
    pooled_grads = tf.reduce_mean(grads,axis=(0,1,2))
    # add a new axis to do matrix multiplcation
    pooled_grads_one = pooled_grads[...,tf.newaxis]

    # multiply the image what we saw by the gradients to up them
    heatmap = last_conv_layer_output @ pooled_grads_one
    # get rid of the negitives
    heatmap = tf.maximum(heatmap,0)
    # get rid of the one axises
    heatmap = tf.squeeze(heatmap)

    # normlize the numbers to 0-1
    # add the small number incase the whole thing was 0 so we dont divde by zero
    heatmap /= heatmap.reduce_max(heatmap) + 1e-8

    # return the heatmap numbers as a numpy so openCv can work with it.
    return heatmap.numpy()

def build_model(name: str, img_size):
    name = name.lower()
    if name == "a":
        return make_modelA(img_size=img_size)
    # elif name == "b":
    #     return make_modelB(img_size=img_size)
    else:
        raise ValueError(f"Unknown model: {name}")