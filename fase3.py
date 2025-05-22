from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

# ─── Rutas y parámetros necesarios ─────────────────────────────────────
target_size = (512, 512)
train_dir = r'C:\Develop\IA\cancer_de_mama\dataset\cancer_de_mama_edit_3\train'
valid_dir = r'C:\Develop\IA\cancer_de_mama\dataset\cancer_de_mama_edit_3\valid'
test_dir  = r'C:\Develop\IA\cancer_de_mama\dataset\cancer_de_mama_edit_3\test'

batch_size = 16
seed = 42


# ─── Reconstruir arquitectura EXACTA ─────────────────────────────────
backbone = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(*target_size, 3))

# Congelar parcialmente como hiciste en fase 1
start_training = False
for layer in backbone.layers:
    if layer.name == 'block7a_expand_conv':
        start_training = True
    layer.trainable = start_training

inp = layers.Input(shape=(*target_size, 3))
x = backbone(inp, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu', name='feature_dense')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(2, activation='softmax')(x)

model = models.Model(inp, out)

# Compilar antes de cargar pesos (por capas como BatchNorm)
model.compile(
    optimizer=Adam(1e-2),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Recall(name='recall_pos'), keras.metrics.AUC(name='auc')]
)

# ─── Cargar los pesos entrenados en fase 1 ──────────────────────────
model.load_weights(r'C:\Develop\IA\cancer_de_mama\models\last_fase1_v3.weights.h5')
print("✅ Pesos de Fase 1 cargados correctamente.")

# ─── Crear extractor desde la capa que te interese ──────────────────────────
feature_extractor = Model(
    inputs=model.input,
    outputs=model.get_layer('feature_dense').output  # el nombre de la capa densa antes del softmax
)

# ─── Data generator sin shuffle para obtener correspondencia perfecta ───────
datagen = ImageDataGenerator(
    preprocessing_function=keras.applications.efficientnet.preprocess_input
)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=target_size, batch_size=batch_size,
    class_mode='sparse', shuffle=False, seed=seed
)
valid_gen = datagen.flow_from_directory(
    valid_dir, target_size=target_size, batch_size=batch_size,
    class_mode='sparse', shuffle=False
)
test_gen  = datagen.flow_from_directory(
    test_dir,  target_size=target_size, batch_size=batch_size,
    class_mode='sparse', shuffle=False
)

# ─── Extraer embeddings ─────────────────────────────────────────────────────
features_train = feature_extractor.predict(train_gen, verbose=1)
y_train        = train_gen.classes

features_valid = feature_extractor.predict(valid_gen, verbose=1)
y_valid        = valid_gen.classes

features_test  = feature_extractor.predict(test_gen, verbose=1)
y_test         = test_gen.classes

# ─── Entrenar y evaluar SVM ─────────────────────────────────────────────────
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(features_train, y_train)

valid_pred = svm.predict(features_valid)
valid_acc = accuracy_score(y_valid, valid_pred)
print(f"✅ Validación SVM Accuracy: {valid_acc:.4f}")

test_pred = svm.predict(features_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"🎯 Test Accuracy SVM: {test_acc:.4f}")
joblib.dump(svm, 'svm_classifier_v3.joblib')