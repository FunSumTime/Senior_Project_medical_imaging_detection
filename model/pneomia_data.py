import tensorflow as tf
import pathlib
data_dir = pathlib.Path("pneomia")

Batch_size = 12

# will walk the folder tree and go to the directory and label them for classification

train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir / "train",
    label_mode="binary",
    image_size=(256,256),
    batch_size=Batch_size,
    shuffle=True
)


test_data = tf.keras.utils.image_dataset_from_directory(
    data_dir / "test",
    label_mode="binary",
    image_size=(256,256),
    batch_size=Batch_size,
    shuffle=False
)


data_bank = {
    0: train_data,
    1: test_data
}

def grab_data(num):
    print(data_bank[0].class_names)
    if num == 1:
        return data_bank[1]
    else:
        return data_bank[0]
    
print(train_data.class_names)