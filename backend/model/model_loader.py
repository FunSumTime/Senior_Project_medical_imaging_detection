import tensorflow as tf
def load_model(path):

    Model = tf.keras.models.load_model(path,safe_mode=False)
    return Model
