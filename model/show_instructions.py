print("instructions for run_pneomia.py arguments")
#   p.add_argument("--img_size", type=int, default=256)
#     p.add_argument("--epochs", type=int, default=10)
#     p.add_argument("--model", type=str, default="a")

#     p.add_argument("--save", type=int, default=1)
#     p.add_argument("--save_path", type=str, default="pnemonia_models")
#     p.add_argument("--model_file", type=str, default="model_pnemonia.keras")  # include extension
#     p.add_argument("--threshold_auc", type=float, default=0.80)         # nice to control
#     p.add_argument("--action", type=str, default="do-fit")
#     p.add_argument("--gradcam_img", type=str, default=None)
#     p.add_argument("--gradcam_out", type=str, default="gradcam_overlay.png")

print("--img_size to adjust image size, default 256")
print("--epochs to adjust how many epochs the model will go through, defualt 10")
print("--model to choose what model to make only one for right now, defualt 'a'")
print("--save to decide weither to save the current training of the model or not default 1(True)")
print("--save_path where to save the model to, defualt 'pnemonia_models'")
print("--model_file what to have the model be called in the saved file, defualt, model_pnemonia.keras")
print("--threshold_auc accuracy threshold on to weither or not to save the model, defualt .80")
print("--action var to say what to do in this run (do-fit do a training of a model, gradcam-one test the grad cam, plot-history show a models history), defualt do-fit")
print("--gradcam_img the image the gradcam-one will use to test, defualt None")
print("--gradcam_out name for the image the gradcam gives out, 'defualt gradcam_overlay.png'")