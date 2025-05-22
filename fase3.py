from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

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