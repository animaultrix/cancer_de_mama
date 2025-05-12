import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("✅ TensorFlow está usando la GPU")
else:
    print("⚠ TensorFlow está usando SOLO CPU")

