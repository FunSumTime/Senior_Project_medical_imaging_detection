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

def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # 1. Create the sub-model we discussed
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Record the gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # For binary, preds is just the score (e.g., 0.95)
        # If we had multiple classes, we'd pick the specific class index here
        class_channel = preds[:, 0]

    # 3. This is the "magic" step: calculate gradients of the score 
    # with respect to the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    return grads, last_conv_layer_output
def build_model(name: str, img_size):
    name = name.lower()
    if name == "a":
        return make_modelA(img_size=img_size)
    # elif name == "b":
    #     return make_modelB(img_size=img_size)
    else:
        raise ValueError(f"Unknown model: {name}")