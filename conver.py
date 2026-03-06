import tensorflow as tf

model = tf.keras.models.load_model(
    "models/resnet50_cancer_model-MRI-finetuned-version-1.keras",
    compile=False
)

model.save("mri_model.h5")

print("Model converted successfully")