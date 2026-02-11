import tensorflow as tf
import cv2
import numpy as np

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



# function to make the grad model that will route the last layer and the outputs to  it
def make_grad_model(model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    return grad_model


# make the heatmap 
def grad_engine(img_array, grad_model):
    with tf.GradientTape() as tape:

        # grab the outputs from the sub model, as the gradient wants to do the gradent with respect to the map based of the prediction
        last_conv_layer_output, predictions = grad_model(img_array,training=False)
        tape.watch(last_conv_layer_output)

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
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)


    # return the heatmap numbers as a numpy so openCv can work with it.
    return heatmap.numpy()

# take the heatmap and the orginal image and display the heatmap
def display_gradcam(original_rgb, heatmap):
    h, w = original_rgb.shape[:2]
    # resize it to 256*256
    heatmap_resized = cv2.resize(heatmap, (w, h))
    # make them from 0 - 255
    heatmap_8bit = np.uint8(255 * heatmap_resized)
    # add teh color scheme
    color_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

    # make the orginal to formate
    img_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    # times and add them to gether
    blended = cv2.addWeighted(color_heatmap, 0.4, img_bgr, 0.6, 0)

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)




def build_model(name: str, img_size):
    name = name.lower()
    if name == "a":
        return make_modelA(img_size=img_size)
    # elif name == "b":
    #     return make_modelB(img_size=img_size)
    else:
        raise ValueError(f"Unknown model: {name}")
    

def get_img_array(img_path, size=(256, 256)):
    img = tf.keras.utils.load_img(img_path, target_size=size, color_mode="rgb")
    array = tf.keras.utils.img_to_array(img)  # (H,W,3) float32
    array = np.expand_dims(array, axis=0)     # (1,H,W,3)
    return array

    

#make_grad_model: Creates a multi-output model to "spy" on the internal features. 

# grad_engine: Uses the tape to find which features moved the needle on the prediction. 

# display_gradcam: Resizes, colors, and blends that info back onto the original X-ray. 
def test_overlay_heatmap(model, img_path):
    # get grad model
    grad_model = make_grad_model(model, "last_conv_layer")

    # get the image
    img_array = get_img_array(img_path, size=(256,256))
    # get the heatmap
    heatmap = grad_engine(img_array, grad_model)

    # make a orginal image
    original_rgb = img_array[0].astype("uint8")
    # get the overlay
    overlay = display_gradcam(original_rgb, heatmap)

    print("heatmap:", heatmap.shape, heatmap.min(), heatmap.max())
    return overlay



