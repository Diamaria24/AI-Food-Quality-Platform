from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from scripts.training.mobilenet_loader import (
    load_datasets
)

_,_,test_data = load_datasets()

model = load_model(
    "models/trained/mobilenet.keras"
)

y_true = test_data.classes

y_pred = model.predict(
    test_data
)

n_classes = 8

y_true_bin = label_binarize(

    y_true,

    classes=range(n_classes)
)

plt.figure(figsize=(10,8))

for i in range(n_classes):

    fpr,tpr,_ = roc_curve(

        y_true_bin[:,i],

        y_pred[:,i]
    )

    roc_auc = auc(

        fpr,

        tpr
    )

    plt.plot(

        fpr,

        tpr,

        label=f"Class {i} AUC={roc_auc:.3f}"
    )

plt.plot(

    [0,1],

    [0,1],

    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves"
)

plt.legend()

plt.savefig(

    "outputs/plots/roc_curve.png",

    dpi=300
)

plt.show()

