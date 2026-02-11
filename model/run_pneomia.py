import argparse
import os
import tensorflow as tf
import joblib

from pneomia_data import grab_data
from pneomia_model import build_model
from model_history import plot_history

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--model", type=str, default="a")

    p.add_argument("--save", type=int, default=1)
    p.add_argument("--save_path", type=str, default="pnemonia_models")
    p.add_argument("--model_file", type=str, default="model_pnemonia.keras")  # include extension
    p.add_argument("--threshold_auc", type=float, default=0.80)         # nice to control
    p.add_argument("--action", type=str, default="do-fit")
    p.add_argument("--gradcam_img", type=str, default=None)
    p.add_argument("--gradcam_out", type=str, default="gradcam_overlay.png")
    return p.parse_args()




def main():
    args = parse_args()

    if args.action == "plot-history":
        plot_history(args)
        return
    img_size = (args.img_size, args.img_size)

    train_ds = grab_data(0)
    test_ds  = grab_data(1)

    # TEMP: using test as validation (okay for now, but later split train into val)
    val_ds = test_ds


    # create the model
    model = build_model(args.model, img_size=img_size)
    # compile it, what things it should use
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()
    if args.action == "gradcam-one":
        from pneomia_model import make_grad_model, grad_engine, display_gradcam, get_img_array  # or wherever you put them
        import cv2

        # Build the model graph (important sometimes)
        model(tf.zeros((1, args.img_size, args.img_size, 3)))

        # Make grad model + run on one image
        grad_model = make_grad_model(model, "last_conv_layer")
        img_array = get_img_array(args.gradcam_img, size=(args.img_size, args.img_size))
        heatmap = grad_engine(img_array, grad_model)

        original_rgb = img_array[0].astype("uint8")
        overlay = display_gradcam(original_rgb, heatmap)

        # Save
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.gradcam_out, overlay_bgr)
        print("Saved:", args.gradcam_out)
        return

    print("hello")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=[early_stop],
    )


    # save history next to the model (or wherever you want)
    os.makedirs(args.save_path, exist_ok=True)
    model_file = os.path.join(args.save_path, args.model_file)  # same base name you use for model
    joblib.dump(history.history, model_file + ".history")

    # print("Saved history ->", model_file + ".history")


    # Decide whether to save AFTER training
    eval_results = model.evaluate(val_ds, verbose=0)
    eval_metrics = dict(zip(model.metrics_names, eval_results))

    # print("Eval metrics:", eval_metrics)

    auc_value = eval_metrics.get("auc")  # because you named it "auc"

    if args.save == 1 and auc_value is not None and auc_value >= args.threshold_auc:
        os.makedirs(args.save_path, exist_ok=True)
        full_path = os.path.join(args.save_path, args.model_file)
        model.save(full_path)
        print(f"Saved model ✅ -> {full_path} (auc={auc_value:.3f})")
    else:
        print(f"Did not save model (auc={auc_value}, threshold={args.threshold_auc}).")

    # If you insist on a final test report, right now it's the same as val_ds.
    # Once you create a real val_ds, evaluate test_ds here for the true final score.
    test_results = model.evaluate(test_ds, verbose=0)
    test_metrics = dict(zip(model.metrics_names, test_results))
    print("Test results:", test_metrics)

if __name__ == "__main__":
    main()
