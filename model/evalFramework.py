import numpy as np

def get_probs_and_labels(model, ds):
    probs_list = []
    labels_list = []

    for x, y in ds:
        p = model.predict(x, verbose=0).reshape(-1)   # sigmoid outputs
        probs_list.append(p)
        labels_list.append(y.numpy().reshape(-1))

    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list).astype(int)
    return probs, labels


def confusion_at_threshold(probs, labels, thresh=0.5):
    preds = (probs >= thresh).astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return acc, precision, recall, f1


def eval_report(model, ds, thresh=0.5):
    probs, labels = get_probs_and_labels(model, ds)
    tp, tn, fp, fn = confusion_at_threshold(probs, labels, thresh)
    acc, prec, rec, f1 = metrics_from_confusion(tp, tn, fp, fn)

    print(f"threshold={thresh}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    return probs, labels


# probs, labels = eval_report(model, test_ds, thresh=0.5)

# This lets us:

# change threshold

# compute custom metrics

# plot ROC curve

# analyze confidence behavior


# If the model predicts pneumonia a lot:

# Recall high

# Precision may drop

# If it predicts cautiously:

# Precision high

# Recall may drop