import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# --- 1. CONFIGURATION ---
TRAIN_PATH = 'OneDrive_1_09-04-2026/x_train/'
TEST_PATH = 'OneDrive_1_09-04-2026/x_test/'
Y_TRAIN_FILE = 'OneDrive_1_09-04-2026/y_train_v2.csv'

# --- 2. CHARGEMENT DES LABELS ---
labels_df = pd.read_csv(Y_TRAIN_FILE)

def load_images_from_folder(folder, start_idx, end_idx, is_train=True):
    images = []
    labels = []
    total = end_idx - start_idx + 1

    print(f"--- Chargement de {total} images dans {folder} ---")

    for count, i in enumerate(range(start_idx, end_idx + 1)):
        img_name = f"img_{i}.png"
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)

            if is_train:
                target = labels_df.loc[labels_df['id'] == (i - 1), 'target'].values[0]
                labels.append(target)

        if (count + 1) % 500 == 0:
            print(f"Progression : {count + 1} / {total} images chargées...")

    return np.array(images), np.array(labels)

# Chargement Train & Test
X_train, y_train = load_images_from_folder(TRAIN_PATH, 1, 9227, is_train=True)
X_kaggle, _ = load_images_from_folder(TEST_PATH, 9228, 13182, is_train=False)

IMG_H, IMG_W, IMG_C = X_train[0].shape

# Normalisation
X_train = X_train.astype('float32') / 255.0
X_kaggle = X_kaggle.astype('float32') / 255.0
y_train_cat = to_categorical(y_train, num_classes=4)

# --- 3. ARCHITECTURE DU MODÈLE (CNN) ---
model = Sequential([
    Input(shape=(IMG_H, IMG_W, IMG_C)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# --- 4. ENTRAÎNEMENT ---
print("\nDébut de l'entraînement...")
history = model.fit(X_train, y_train_cat, epochs=30, batch_size=32)

# --- 5. PRÉDICTION & KAGGLE SUBMISSION ---
print("\nGénération des prédictions pour Kaggle...")
preds = model.predict(X_kaggle)
final_preds = np.argmax(preds, axis=1)

submission = pd.DataFrame({
    'id': np.arange(9227, 9227 + len(final_preds)),
    'target': final_preds
})

submission.to_csv('my_submission_v5.csv', index=False)
print("Fichier 'my_submission_v5.csv' prêt !")

# --- 6. COURBES D'ENTRAÎNEMENT ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title("Évolution de l'Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Évolution de la Loss')
plt.legend()
plt.show()

np.save('val_preds.npy', np.argmax(model.predict(X_train[int(len(X_train)*0.8):]), axis=1))
np.save('val_true.npy', np.argmax(y_train_cat[int(len(y_train_cat)*0.8):], axis=1))