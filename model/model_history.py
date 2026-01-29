# !/usr/bin/env python3

#
# Display the model history.
#

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_history(my_args):
    """
    Plot the history of the model training.
    
    Assumes model_file.history has the fit history.
    Assumes that the there are equal number of training and validation values.
    """


    history_path = os.path.join(
        my_args.save_path,
        my_args.model_file + ".history"
    )
    history = joblib.load(history_path)

    epochs = len(history["loss"])
    learning_curve_filename = "{}.learning_curve.png".format(my_args.model_file)

    #
    # Display the learning curves
    #
    line_count = len(history.keys())
    if line_count == 2:
        line_style = ["r--+", 
                      "b-+"]
    elif line_count == 4:
        line_style = ["r--*", "r--+", 
                      "b-*", "b-+"]
    elif line_count == 6:
        line_style = ["r--", "r--*", "r--+", 
                      "b-", "b-*", "b-+"]
    elif line_count == 8:
        line_style = ["r--", "r--*", "r--+", "r--x", 
                      "b-", "b-*", "b-+", "b-x"]
    elif line_count == 10:
        line_style = ["r--", "r--*", "r--+", "r--x", "r--1", 
                      "b-", "b-*", "b-+", "b-x", "b-1"]
    else:
        raise Exception("Invalid line count: {}".format(line_count))

    pd.DataFrame(history).plot(
        figsize=(8, 5), xlim=[0, epochs-1], grid=True, xlabel="Epoch",
        style=line_style)
    # plt.show()
    plt.savefig(learning_curve_filename)
    plt.clf()
    return
