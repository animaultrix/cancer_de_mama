import numpy as np
import os
import datetime 
import tensorflow as tf
from tensorflow import keras
import datetime
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.utils.class_weight import compute_class_weight


# ─── Parámetros y rutas ─────────────────────────────────────────────
version = "2"
target_size = (512, 512)       # originales son de 640, 640
batch_size  = 16  # ↓ para no saturar GPU
seed        = 42

base_dir    = f'/dataset/cancer_de_mama_edit_3'
train_dir   = os.path.join(base_dir, 'train')
valid_dir   = os.path.join(base_dir, 'valid')
test_dir    = os.path.join(base_dir, 'test')

# ────────────────────────────────────────────────────────────
#  3. DATA AUGMENTATION
# ────────────────────────────────────────────────────────────
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
    class_mode='binary', shuffle=True, seed=seed)                 # <<< ‘binary’
valid_gen = valid_datagen.flow_from_directory(
    valid_dir, target_size=target_size, batch_size=batch_size,
    class_mode='binary', shuffle=False)
test_gen = test_datagen.flow_from_directory(
    test_dir,  target_size=target_size, batch_size=batch_size,
    class_mode='binary', shuffle=False)


# ────────────────────────────────────────────────────────────
#  4. CLASS WEIGHTS  (equilibra la penalización)
# ────────────────────────────────────────────────────────────
y_train = train_gen.classes
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))      # {0: w0, 1: w1}
print("Class weights:", class_weights)


# ─── 5. MODELO (EfficientNetB3) ─────────────────────────────
backbone = EfficientNetB3(weights='imagenet', include_top=False,
                          input_shape=(*target_size, 3))
backbone.trainable = False

inp = layers.Input(shape=(*target_size,3))
x = backbone(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inp, out)

model.compile(
    optimizer=Adam(1e-4),
    loss=BinaryFocalCrossentropy(gamma=2.0, alpha=0.75),
    metrics=['accuracy',
             keras.metrics.Recall(name='recall_pos'),
             keras.metrics.AUC(name='auc')]
)

# ─── 6. CALLBACKS ───────────────────────────────────────────
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
cbs1 = [EarlyStopping('val_loss', patience=8, restore_best_weights=True),
       ReduceLROnPlateau('val_loss', factor=0.5, patience=6, min_lr=1e-6),
       ModelCheckpoint(f'models/effb3_best_v{version}.keras',
                       monitor='val_loss', save_best_only=True),
       TensorBoard(log_dir=logdir, histogram_freq=1)]
cbs2 = [EarlyStopping('val_loss', patience=8, restore_best_weights=True),
       ReduceLROnPlateau('val_loss', factor=0.5, patience=6, min_lr=1e-6),
       ModelCheckpoint(f'models/best_v{version}.keras',
                       monitor='val_loss', save_best_only=True),
       TensorBoard(log_dir=logdir, histogram_freq=1)]

# ─── 7. FASE 1 (cabeza) ────────────────────────────────────
model.fit(train_gen, validation_data=valid_gen,
          epochs=15, class_weight=class_weights, callbacks=cbs1,
          verbose=1)
model.save('models/last_fase1.keras')
# ─── Evaluación final ──────────────────────────────────────────────
loss, acc, recall, auc = model.evaluate(test_gen, verbose=0)
print(f"\nTest → Acc: {acc:.3f} | Recall (+): {recall:.3f} | AUC: {auc:.3f}")
# ─── 8. FASE 2 (descongelar 150 capas) ─────────────────────
for layer in backbone.layers[-150:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss=BinaryFocalCrossentropy(gamma=2.0, alpha=0.75),
    metrics=['accuracy',
             keras.metrics.Recall(name='recall_pos', threshold=0.5),
             keras.metrics.AUC(name='auc')]
)

model.fit(train_gen, validation_data=valid_gen,
          epochs=30, class_weight=class_weights, callbacks=cbs2,
          verbose=1)
model.save('models/last_fase2.keras')

# ─── 9. EVALUACIÓN ─────────────────────────────────────────
loss, acc, rec, auc = model.evaluate(test_gen, verbose=0)
print(f"\nTest → Acc: {acc:.3f} | Recall (+): {recall:.3f} | AUC: {auc:.3f}")