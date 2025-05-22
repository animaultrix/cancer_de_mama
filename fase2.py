import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ─── Parámetros ─────────────────────────────────────────────────────
version = "3"
target_size = (512, 512)
batch_size = 16
seed = 42

base_dir = f'/dataset/cancer_de_mama_edit_3'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# ─── Data generators ────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=keras.applications.efficientnet.preprocess_input,
    rotation_range=5,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='reflect'
)
valid_datagen = ImageDataGenerator(preprocessing_function=keras.applications.efficientnet.preprocess_input)
test_datagen  = ImageDataGenerator(preprocessing_function=keras.applications.efficientnet.preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=target_size, batch_size=batch_size,
    class_mode='binary', shuffle=True, seed=seed, color_mode='rgb')                 # <<< ‘binary’
valid_gen = valid_datagen.flow_from_directory(
    valid_dir, target_size=target_size, batch_size=batch_size,
    class_mode='binary', shuffle=False,
    color_mode='rgb')
test_gen = test_datagen.flow_from_directory(
    test_dir,  target_size=target_size, batch_size=batch_size,
    class_mode='binary', shuffle=False,
    color_mode='rgb')


# ─── Class weights ─────────────────────────────────────────────────
y_train = train_gen.classes
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

# ─── Cargar modelo entrenado en fase 1 ──────────────────────────────
model = keras.models.load_model('models/last_fase1.keras', compile=False)

# ─── Descongelar últimas 150 capas ──────────────────────────────────
backbone = model.layers[1]
for layer in backbone.layers[-150:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss=BinaryFocalCrossentropy(gamma=2.0, alpha=0.75),
    metrics=['accuracy', keras.metrics.Recall(name='recall_pos', threshold=0.5), keras.metrics.AUC(name='auc')]
)

# ─── Callbacks fase 2 ───────────────────────────────────────────────
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
cbs2 = [
    EarlyStopping('val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau('val_loss', factor=0.5, patience=6, min_lr=1e-6),
    ModelCheckpoint(f'models/best_v{version}.keras', monitor='val_loss', save_best_only=True),
    TensorBoard(log_dir=logdir, histogram_freq=1)
]

# ─── Entrenamiento Fase 2 ───────────────────────────────────────────
model.fit(train_gen, validation_data=valid_gen, epochs=30,
          class_weight=class_weights, callbacks=cbs2, verbose=1)

model.save('models/last_fase2.keras')

# ─── Evaluación final ───────────────────────────────────────────────
loss, acc, recall, auc = model.evaluate(test_gen, verbose=0)
print(f"\nTest → Acc: {acc:.3f} | Recall (+): {recall:.3f} | AUC: {auc:.3f}")
# fase2