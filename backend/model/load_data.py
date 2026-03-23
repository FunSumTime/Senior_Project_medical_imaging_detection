import tensorflow as tf
from preprocess import center_crop, mask_corners
DATA_ROOT = "/pneomia"
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 1337
# to get the same shuffel

# split the train into 80:20

# make two dataset pipelines
# this would be for loading

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_ROOT + "/train",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_ROOT + "/train",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int"
)




train_ds2 = train_ds.map(center_crop).prefetch(tf.data.AUTOTUNE)
val_ds2   = val_ds.map(center_crop).prefetch(tf.data.AUTOTUNE)





train_ds_mask = train_ds.map(lambda x,y: mask_corners(x,y, frac=0.12, top_frac=0.10)).prefetch(tf.data.AUTOTUNE)
val_ds_mask   = val_ds.map(lambda x,y: mask_corners(x,y, frac=0.12, top_frac=0.10)).prefetch(tf.data.AUTOTUNE)


class_names = train_ds.class_names
print("Classes: ", class_names)