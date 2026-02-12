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


for images, labels in train_data.take(1):
    print(labels.shape)

def grab_data(num):
    print(data_bank[0].class_names)
    if num == 1:
        return data_bank[1]
    else:
        return data_bank[0]
    
print(train_data.class_names)


import numpy as np

def count_labels(ds):
    n0 = 0
    n1 = 0
    total = 0

    for _, y in ds:
        y = y.numpy().reshape(-1)  # batch -> 1D
        n1 += int(np.sum(y))
        n0 += int(np.sum(1 - y))
        total += len(y)

    print(f"total={total}  normal(0)={n0}  pneumonia(1)={n1}  ratio pneumonia={n1/total:.3f}")
    return n0, n1

count_labels(train_data)
count_labels(test_data)