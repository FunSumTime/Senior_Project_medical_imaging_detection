import tensorflow as tf

def make_modelA(img_size=(256,256)):
    # what the model will recive 256X256 ,3 saying my model will recive these tensors

    inputs = tf.keras.Input(shape=img_size + (3,))

    # start pipeline from inputs
    x = inputs

    # (recommended) normalize to 0..1
    # rescaling because they like numbers in scale of 0-1
    x = tf.keras.layers.Rescaling(1./255)(x)

    # augmentation (training only automatically)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    # 32 filters size 3*3
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    # reduce size and keep strongest inputs
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # the classify into two groups
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model



def build_model(name: str, img_size):
    name = name.lower()
    if name == "a":
        return make_modelA(img_size=img_size)
    # elif name == "b":
    #     return make_modelB(img_size=img_size)
    else:
        raise ValueError(f"Unknown model: {name}")