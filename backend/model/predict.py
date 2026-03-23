import numpy as np
from .load_data import val_ds2
from .model_loader import load_model

model_cam = load_model("/models/model_cam.keras")

y_true= []
y_pred_probs = []
for images, labels in val_ds2:
  preds = model_cam.predict(images,verbose=0)
  # model takes in batches and outputs the probablits
  y_pred_probs.extend(preds)
  # preds is a batch of predictions arrays
  # extend means take each one and add it indivdually
  y_true.extend(labels.numpy())

# make them easier
# so what we do is take in a batch predict move it into a array and compare
y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

# argmax gives index of largest value
# so it turns into 0,1,0,1...
y_pred = np.argmax(y_pred_probs, axis=1)
print("done")

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true,y_pred)
print("confusion matrix")
print(cm)
# TN FP
# FN TP
TN = cm[0][0]
FP = cm[0][1]

FN = cm[1][0]
TP = cm[1][1]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
print("precision: ", precision)
print("recall: ", recall)


F1 = 2* (precision * recall) / (precision + recall)
print("F1: ", F1)



p_pneumonia = y_pred_probs[:,1]
def metrics_at_threshold(t):
  y_hat = (p_pneumonia >= t).astype(int)
  tn,fp,fn,tp = confusion_matrix(y_true,y_hat).ravel()

  precision = tp /(tp+fp + 1e-9)
  recall = tp / (tp + fn + 1e-9)
  f1 = 2*precision * recall / (precision + recall + 1e-9)

  return tn,fp,fn,tp,precision,recall,f1

# .1 high thresshold will catcth more and .5 is normal
for t in [0.1,0.2,0.3,0.4,0.5]:
  tn,fp,fn,tp,prec,rec,f1 = metrics_at_threshold(t)
  print(f"t={t:.1f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  FP={fp} FN={fn}")