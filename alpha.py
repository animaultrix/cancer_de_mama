import numpy as np
import os
import datetime
import tensorflow as tf
from tensorflow import keras
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


print("TensorFlow:", tf.__version__)

# ╔═══════════════════════════════════════╗
# 2) RUTAS (🎯 AJUSTA AQUÍ)
# ╚═══════════════════════════════════════╝
BASE = pathlib.Path(r"C:\Develop\IA\alpha")       # carpeta raíz
SRC_TRAIN = BASE / "asl_alphabet_train"           # A/,B/,C/… nothing/, space/
SRC_TEST  = BASE / "asl_alphabet_test"            # opcional (no usado aquí)

RUN_NAME   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
WORK_DIR   = BASE / f"runs\cnn_{RUN_NAME}"
MODEL_BEST = WORK_DIR / "asl_cnn_best.keras"
MODEL_LAST = WORK_DIR / "asl_cnn_last.keras"
LOG_DIR    = WORK_DIR / "logs"

WORK_DIR.mkdir(parents=True, exist_ok=True)

# ╔═══════════════════════════════════════╗
# 3) PARAMETROS
# ╚═══════════════════════════════════════╝
IMG_SIZE = (96, 96)          # resolución adecuada para CNN ligera
BATCH    = 64                # ajusta según tu GPU / RAM
SEED     = 42
EPOCHS   = 40

# ╔═══════════════════════════════════════╗
# 4) GENERADORES CON AUGMENTACIÓN
# ╚═══════════════════════════════════════╝
augmenter = ImageDataGenerator(
    rescale=1./255,
    #rotation_range=15,
    #zoom_range=0.1,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #shear_range=10,
    validation_split=0.2,
    #fill_mode="reflect"
)

train_gen = augmenter.flow_from_directory(
    SRC_TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='training', seed=SEED
)
val_gen = augmenter.flow_from_directory(
    SRC_TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='validation', seed=SEED
)

num_classes = train_gen.num_classes
print(f"\nTrain imgs: {train_gen.n} | Val imgs: {val_gen.n} | Clases: {num_classes}")


# ╔═══════════════════════════════════════╗
# 6) DEFINICIÓN DEL MODELO CNN
# ╚═══════════════════════════════════════╝
def build_cnn(input_shape, num_classes):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPool2D(),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPool2D(),

        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPool2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_cnn(IMG_SIZE + (3,), num_classes)
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ╔═══════════════════════════════════════╗
# 7) CALLBACKS
# ╚═══════════════════════════════════════╝
cbs = [
    callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    callbacks.ModelCheckpoint(MODEL_BEST, save_best_only=True, verbose=1)
]

# ╔═══════════════════════════════════════╗
# 8) ENTRENAMIENTO
# ╚═══════════════════════════════════════╝
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=cbs
)

model.save(MODEL_LAST)
print("✅ Modelo final guardado en", MODEL_LAST)

# ╔═══════════════════════════════════════╗
# 9) CURVAS DE ENTRENAMIENTO
# ╚═══════════════════════════════════════╝
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(WORK_DIR / "history.png")
print("📈 Gráfica guardada en", WORK_DIR / "history.png")