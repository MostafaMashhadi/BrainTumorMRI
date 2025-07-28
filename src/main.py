import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image
import tf2onnx
import coremltools as ct
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
import random

# Configure GPU (Mac M3 specific setup)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU configured successfully:", physical_devices)
    except Exception as e:
        print(f"Error setting GPU config: {e}")

# Parameters
IMAGE_SIZE = (256, 256)  # Image resolution
DATA_DIR = "/Users/mostafamashhadizadeh/Desktop/MyProjects/BrainTumorMRI/Data"
BATCH_SIZE = 64 
CATEGORIES = os.listdir(os.path.join(DATA_DIR, "Train"))
if ".DS_Store" in CATEGORIES:
    CATEGORIES.remove(".DS_Store")

# Load dataset function
def load_dataset(folder):
    images, labels = [], []
    for label in CATEGORIES:
        folder_path = os.path.join(DATA_DIR, folder, label)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        for file in tqdm(image_files, desc=f"{folder}/{label}"):
            try:
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img) / 255.0
                images.append(img_array[..., np.newaxis])
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load training and testing data
X_train, y_train = load_dataset("Train")
X_test, y_test = load_dataset("Test")

# Encode labels to one-hot format
encoder = LabelEncoder()
y_train_enc = to_categorical(encoder.fit_transform(y_train))
y_test_enc = to_categorical(encoder.transform(y_test))

# Print dataset shapes for verification
print("X_train shape:", X_train.shape)
print("y_train_enc shape:", y_train_enc.shape)
print("X_test shape:", X_test.shape)
print("y_test_enc shape:", y_test_enc.shape)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CATEGORIES), activation='softmax')
])

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=0.0001)

# Train the model
history = model.fit(
    X_train, y_train_enc,
    batch_size=BATCH_SIZE,
    epochs=17,
    validation_data=(X_test, y_test_enc),
    callbacks=[early_stopping, reduce_lr]
)

# Save model in H5 format
try:
    model.save("BrainTumorClassifier.h5")
    print("✅ Model saved as 'BrainTumorClassifier.h5'")
except Exception as e:
    print(f"Error saving TensorFlow model: {e}")

# Convert and save the model in ONNX format
try:
    spec = tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec([None, 256, 256, 1], tf.float32, name="conv2d_input")], output_path="BrainTumorClassifier.onnx")
    print("✅ Model saved as 'BrainTumorClassifier.onnx'")
except Exception as e:
    print(f"Error converting to ONNX: {e}")

# Convert and save the model in Core ML format
try:
    CORRECT_CATEGORIES = ["notumor", "glioma", "meningioma", "pituitary"]
    coreml_model = ct.convert(
        model,
        source='tensorflow',
        inputs=[ct.ImageType(name="conv2d_input", shape=(1, 256, 256, 1), scale=1/255.0, channel_first=False, color_layout='G')],
        classifier_config=ct.ClassifierConfig(CORRECT_CATEGORIES),
        compute_precision=ct.precision.FLOAT32
    )
    coreml_model.save('BrainTumorClassifier.mlpackage')
    print("✅ Model saved as 'BrainTumorClassifier.mlpackage'")
except Exception as e:
    print(f"Error converting to Core ML: {e}")

# Create output folder
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Training & Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/accuracy_plot.png")
plt.close()

# Training & Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/loss_plot.png")
plt.close()

# Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_enc, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CORRECT_CATEGORIES)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()

# Sample Predictions (Random 9 examples from test set)
plt.figure(figsize=(12, 12))
indices = random.sample(range(len(X_test)), 9)

for i, idx in enumerate(indices):
    image = X_test[idx]
    true_label = CORRECT_CATEGORIES[y_true[idx]]
    pred_label = CORRECT_CATEGORIES[y_pred[idx]]

    plt.subplot(3, 3, i+1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
    plt.axis('off')

plt.suptitle('Sample Predictions on Test Set', fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/sample_predictions.png")
plt.close()