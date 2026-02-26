import matplotlib.pyplot as plt
from gradcam import make_gradcam_heatmap
from load_data import val_ds,val_ds2,val_ds_mask
from model_loader import load_model
from config import CLASS_NAMES as class_names, IMG_SIZE
import numpy as np

model_cam = load_model("/models/model_cam.keras")
# pull one batch
images, labels = next(iter(val_ds2))
heatmap = make_gradcam_heatmap(images)

plt.figure(figsize=(4,4))
plt.imshow(heatmap)
plt.axis("off")
plt.title("Grad-Cam heatmap")
plt.show()

print("True label:", labels[0].numpy(), "(", class_names[labels[0].numpy()], ")")
pred = model_cam.predict(images[:1], verbose=0)[0]
print("Pred probs:", pred, "->", class_names[int(np.argmax(pred))])


import cv2

# image 0 from the last batch you used
img0 = images[0].numpy()  # shape (224,224,3), values are 0..255-ish from dataset loader

# make sure it's in 0..255 uint8 for OpenCV display
img0_u8 = np.clip(img0, 0, 255).astype(np.uint8)

# resize heatmap from (7,7) -> (224,224)
hm = heatmap
hm_resized = cv2.resize(hm, (IMG_SIZE, IMG_SIZE))
hm_u8 = np.uint8(255 * hm_resized)

# colorize heatmap and overlay
hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)          # BGR
hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)           # RGB

overlay = (0.6 * img0_u8 + 0.4 * hm_color).astype(np.uint8)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(img0_u8); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(hm_resized); plt.title("Heatmap (upsampled)"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")
plt.show()