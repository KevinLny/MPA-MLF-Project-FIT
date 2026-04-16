import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, Input)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- 1. CONFIGURATION ---
IMG_SIZE    = 64        # Même que ta v1, rapide et efficace
BATCH_SIZE  = 32
EPOCHS      = 50       # EarlyStopping stoppera au bon moment
NUM_CLASSES = 4

TRAIN_PATH   = 'OneDrive_1_09-04-2026/x_train/'
TEST_PATH    = 'OneDrive_1_09-04-2026/x_test/'
Y_TRAIN_FILE = 'OneDrive_1_09-04-2026/y_train_v2.csv'

# --- 2. CHARGEMENT DES LABELS ---
labels_df = pd.read_csv(Y_TRAIN_FILE)

def load_images_from_folder(folder, start_idx, end_idx, is_train=True):
    images = []
    labels = []
    total  = end_idx - start_idx + 1

    print(f"--- Chargement de {total} images dans {folder} ---")

    for count, i in enumerate(range(start_idx, end_idx + 1)):
        img_path = os.path.join(folder, f"img_{i}.png")
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)

            if is_train:
                target = labels_df.loc[labels_df['id'] == (i - 1), 'target'].values[0]
                labels.append(target)

        if (count + 1) % 500 == 0:
            print(f"  Progression : {count + 1} / {total}")

    return np.array(images), np.array(labels)

# Chargement
X_train_raw, y_train_raw = load_images_from_folder(TRAIN_PATH, 1,    9227,  is_train=True)
X_kaggle,    _            = load_images_from_folder(TEST_PATH,  9228, 13182, is_train=False)

# Normalisation simple [0, 1] comme ta v1 qui marchait
X_train_raw = X_train_raw.astype('float32') / 255.0
X_kaggle    = X_kaggle.astype('float32')    / 255.0

y_train_cat = to_categorical(y_train_raw, num_classes=NUM_CLASSES)

# Split 80/20 avec stratify pour équilibrer les classes
X_t, X_v, y_t, y_v = train_test_split(
    X_train_raw, y_train_cat,
    test_size=0.2, random_state=42,
    stratify=y_train_raw  # garantit la bonne répartition des classes
)

# --- 3. DATA AUGMENTATION LÉGÈRE ---
# Légère pour ne pas dénaturer les images (compter des personnes = position compte)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_t)

# --- 4. ARCHITECTURE CNN AMÉLIORÉE (basée sur ta v1) ---
# Même esprit mais plus profond et avec BatchNorm pour stabiliser
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Bloc 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    # Bloc 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    # Bloc 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    # Tête de classification
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 5. CALLBACKS ---
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,  # Récupère automatiquement les meilleurs poids
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# --- 6. ENTRAÎNEMENT ---
print("\n=== Début de l'entraînement ===")
steps_per_epoch = len(X_t) // BATCH_SIZE

history = model.fit(
    datagen.flow(X_t, y_t, batch_size=BATCH_SIZE),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=(X_v, y_v),
    callbacks=[early_stop, reduce_lr]
)

# --- 7. PRÉDICTIONS ---
print("\nGénération des prédictions pour Kaggle...")
preds       = model.predict(X_kaggle, batch_size=BATCH_SIZE)
final_preds = np.argmax(preds, axis=1)

val_preds         = model.predict(X_v, batch_size=BATCH_SIZE)
val_preds_classes = np.argmax(val_preds, axis=1)
y_true            = np.argmax(y_v, axis=1)

# --- 8. SUBMISSION CSV ---
submission = pd.DataFrame({
    'id':     np.arange(9227, 9227 + len(final_preds)),
    'target': final_preds
})
submission.to_csv('my_submission_v3.csv', index=False)
print("Fichier 'my_submission_v3.csv' prêt !")

# --- 9. MÉTRIQUES ---
val_loss, val_acc = model.evaluate(X_v, y_v, verbose=0)
print(f"\n--- RÉSULTATS FINAUX ---")
print(f"Accuracy de validation : {val_acc * 100:.2f} %")
print(f"Loss de validation     : {val_loss:.4f}")

print("\n--- RAPPORT DE CLASSIFICATION ---")
print(classification_report(y_true, val_preds_classes,
                            target_names=['0 Pers', '1 Pers', '2 Pers', '3 Pers']))

print(model.summary())

# --- 10. VISUALISATIONS ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title("Évolution de l'Accuracy")
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title("Évolution de la Loss")
axes[1].legend()

cm = confusion_matrix(y_true, val_preds_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[2])
axes[2].set_title('Matrice de Confusion')
axes[2].set_xlabel('Prédiction')
axes[2].set_ylabel('Vrai Label')

plt.tight_layout()
plt.show()
