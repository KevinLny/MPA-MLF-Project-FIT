import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

# --- 1. CONFIGURATION ---
IMG_SIZE = 64 
TRAIN_PATH = 'OneDrive_1_09-04-2026/x_train/'
TEST_PATH = 'OneDrive_1_09-04-2026/x_test/'
Y_TRAIN_FILE = 'OneDrive_1_09-04-2026/y_train_v2.csv'

# --- 2. CHARGEMENT DES LABELS ---
labels_df = pd.read_csv(Y_TRAIN_FILE)

def load_images_from_folder(folder, start_idx, end_idx, is_train=True):
    images = []
    labels = []
    total = end_idx - start_idx + 1 # Pour l'affichage
    
    print(f"--- Chargement de {total} images dans {folder} ---")
    
    for count, i in enumerate(range(start_idx, end_idx + 1)):
        img_name = f"img_{i}.png"
        img_path = os.path.join(folder, img_name)
        
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            
            if is_train:
                target = labels_df.loc[labels_df['id'] == (i - 1), 'target'].values[0]
                labels.append(target)
        
        # Toute les 500 img un print
        if (count + 1) % 500 == 0:
            print(f"Progression : {count + 1} / {total} images chargées...")
            
    return np.array(images), np.array(labels)

# Chargement Train & Test
X_train, y_train = load_images_from_folder(TRAIN_PATH, 1, 9227, is_train=True)
X_kaggle, _ = load_images_from_folder(TEST_PATH, 9228, 13182, is_train=False)

# Normalisation (0-255 -> 0-1) plus facile a travailler avec des valeurs entre 0 et 1
X_train = X_train.astype('float32') / 255.0
X_kaggle = X_kaggle.astype('float32') / 255.0
y_train_cat = to_categorical(y_train, num_classes=4)

# Split interne pour validation (80/20)
X_t, X_v, y_t, y_v = train_test_split(X_train, y_train_cat, test_size=0.2, random_state=42)

# --- 3. ARCHITECTURE DU MODÈLE (CNN) ---
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),                         # On réduit la taille de l'image par deux en ne gardant que le pixel le plus fort. Cela permet de condenser l'information importante et d'ignorer le bruit de fond.
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),                                  # 2D to 1D une longue liste
    Dense(128, activation='relu'),              # Neurones, ils font le liens entre personne trouvés et le nbr de persnne
    Dropout(0.5),                               # Contre l'overfitting
    Dense(4, activation='softmax')              # 4 sorties pour 4 classes : 0, 1, 2, 3
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. ENTRAÎNEMENT ---
print("\nDébut de l'entraînement...")
history = model.fit(X_t, y_t, epochs=30, batch_size=32, validation_data=(X_v, y_v))

# --- 5. PRÉDICTION & KAGGLE SUBMISSION ---
print("\nGénération des prédictions pour Kaggle...")
preds = model.predict(X_kaggle)
final_preds = np.argmax(preds, axis=1)

# Prédictions sur le set de validation interne
val_preds = model.predict(X_v)
val_preds_classes = np.argmax(val_preds, axis=1)
y_true = np.argmax(y_v, axis=1)

# Création de la matrice
cm = confusion_matrix(y_true, val_preds_classes)

# Création du CSV au format Kaggle
submission = pd.DataFrame({
    'id': np.arange(9227, 9227 + len(final_preds)), # Les IDs de test commencent après le train
    'target': final_preds
})

# Affichage des données du model caractéristique

submission.to_csv('my_submission_v5.csv', index=False)
print("Fichier 'my_submission_v5.csv' prêt !")

print(model.summary())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Matrice de Confusion - Validation Interne')
plt.xlabel('Prédiction')
plt.ylabel('Vrai Label')
plt.show()

# Score globaux

val_loss, val_acc = model.evaluate(X_v, y_v, verbose=0)
print(f"\n--- CARACTÉRISTIQUES DU MODÈLE ---")
print(f"Accuracy de validation : {val_acc*100:.2f} %")
print(f"Loss de validation     : {val_loss:.4f}")

print("\n--- RAPPORT DE CLASSIFICATION DÉTAILLÉ ---")
print(classification_report(y_true, val_preds_classes, target_names=['0 Pers', '1 Pers', '2 Pers', '3 Pers']))

# historique d'entraînement (Courbes de Loss/Acc)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Évolution de l\'Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Évolution de la Loss')
plt.legend()
plt.show()