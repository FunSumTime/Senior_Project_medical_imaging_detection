import tensorflow as tf
def load_model(path):

    Model = tf.keras.models.load_model(path)
    return Model
