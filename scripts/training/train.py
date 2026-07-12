"""
=========================================================
Train Baseline CNN
=========================================================
"""

import json

from tensorflow.keras.callbacks import (

    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint

)

from scripts.training.data_loader import (
    load_datasets
)

from scripts.training.baseline_cnn import (
    build_baseline_cnn
)

from scripts.config.constants import (

    EPOCHS,

    TRAINED_MODELS,

    HISTORY_FOLDER

)

print()

print("=" * 70)
print("Loading Dataset...")
print("=" * 70)

train_generator, validation_generator, test_generator = (

    load_datasets()

)

print()

print("=" * 70)
print("Building Model...")
print("=" * 70)

model = (

    build_baseline_cnn()

)

model.summary()

early_stopping = (

    EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1

    )

)

reduce_lr = (

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-6,

        verbose=1

    )

)

checkpoint = (

    ModelCheckpoint(

        filepath=str(

            TRAINED_MODELS

            / "baseline_cnn.keras"

        ),

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1

    )

)

print()

print("=" * 70)
print("Training Started...")
print("=" * 70)

history = model.fit(

    train_generator,

    validation_data=

        validation_generator,

    epochs=EPOCHS,

    callbacks=[

        early_stopping,

        reduce_lr,

        checkpoint

    ]

)

history_file = (

    HISTORY_FOLDER

    / "baseline_history.json"

)

with open(

    history_file,

    "w"

) as file:

    json.dump(

        history.history,

        file

    )

print()

print("=" * 70)

print(

    "Training Completed Successfully."

)

print(

    f"Model Saved : "

    f"{TRAINED_MODELS}"

)

print(

    f"History Saved : "

    f"{history_file}"

)

print("=" * 70)

