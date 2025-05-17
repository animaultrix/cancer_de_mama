# ╔═══════════════════════════════════════╗
# ║   IMPORTS                             ║
# ╚═══════════════════════════════════════╝
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib  # para guardar modelo

# ╔═══════════════════════════════════════╗
# ║   RUTAS Y PARÁMETROS                  ║
# ╚═══════════════════════════════════════╝
SUBSET_ROOT = Path(r"C:\Develop\IA\alpha\asl_alphabet_train")   # ruta real local
LOCAL_TEST  = Path(r"C:\Develop\IA\alpha\asl_alphabet_test")    # si lo tienes
IMG_SIZE    = (96, 96)
BATCH       = 64
SEED        = 42
MODEL_PATH  = Path(r"C:\Develop\IA\alpha\svm_asl_model.joblib")

# ╔═══════════════════════════════════════╗
# ║   FUNCIÓN AUXILIAR                    ║
# ╚═══════════════════════════════════════╝
def dataset_to_numpy(data_dir):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=False,
        label_mode="int"
    )
    X_parts, y_parts = [], []
    for batch_imgs, batch_labels in ds:
        imgs_np = batch_imgs.numpy() / 255.0
        X_parts.append(imgs_np.reshape(imgs_np.shape[0], -1))  # flatten
        y_parts.append(batch_labels.numpy())

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y, ds.class_names

# ╔═══════════════════════════════════════╗
# ║   CARGA Y SPLIT                       ║
# ╚═══════════════════════════════════════╝
X_all, y_all, class_names = dataset_to_numpy(SUBSET_ROOT)
print(f"🗂 Total imágenes (train+val): {X_all.shape[0]}")
print("📚 Clases:", class_names)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

# ╔═══════════════════════════════════════╗
# ║   SVM CON SKLEARN                     ║
# ╚═══════════════════════════════════════╝
print("🔧 Entrenando modelo SVM...")
model = SVC(kernel='linear', probability=True, verbose=True)
model.fit(X_train, y_train)

# ╔═══════════════════════════════════════╗
# ║   EVALUACIÓN                          ║
# ╚═══════════════════════════════════════╝
y_val_pred = model.predict(X_valid)
val_acc = accuracy_score(y_valid, y_val_pred)
print(f"\n✅ Accuracy en VALIDACIÓN: {val_acc:.4f}")
print(classification_report(y_valid, y_val_pred, target_names=class_names))

# ╔═══════════════════════════════════════╗
# ║   GUARDAR MODELO                     ║
# ╚═══════════════════════════════════════╝
joblib.dump(model, MODEL_PATH)
print(f"💾 Modelo guardado en: {MODEL_PATH}")

# ╔═══════════════════════════════════════╗
# ║   TEST “REAL” (si existe)             ║
# ╚═══════════════════════════════════════╝
if LOCAL_TEST.exists():
    X_test, y_test, _ = dataset_to_numpy(LOCAL_TEST)
    y_test_pred = model.predict(X_test)
    print(f"\n🧪 Accuracy en TEST: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
