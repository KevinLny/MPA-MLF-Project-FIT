import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- 1. CONFIGURATION ---
IMG_SIZE    = 96        # Augmenté (64 -> 96) pour plus de détails
BATCH_SIZE  = 32
EPOCHS_FROZEN   = 15   # Phase 1 : on entraîne seulement la tête
EPOCHS_FINETUNE = 20   # Phase 2 : fine-tuning des dernières couches
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MobileNetV2 attend du RGB
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

# Normalisation MobileNetV2 attend des valeurs dans [-1, 1]
X_train_raw = (X_train_raw.astype('float32') / 127.5) - 1.0
X_kaggle    = (X_kaggle.astype('float32')    / 127.5) - 1.0

y_train_cat = to_categorical(y_train_raw, num_classes=NUM_CLASSES)

# Split 80/20
X_t, X_v, y_t, y_v = train_test_split(
    X_train_raw, y_train_cat,
    test_size=0.2, random_state=42, stratify=y_train_raw
)

# --- 3. DATA AUGMENTATION ---
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.85, 1.15]
)
datagen.fit(X_t)

# --- 4. ARCHITECTURE : MobileNetV2 + Tête personnalisée ---
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Phase 1 : on gèle tout le backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# --- 5. CALLBACKS ---
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# --- 6. PHASE 1 : Entraînement de la tête (backbone gelé) ---
print("\n=== PHASE 1 : Entraînement de la tête (backbone gelé) ===")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

steps_per_epoch = len(X_t) // BATCH_SIZE

history1 = model.fit(
    datagen.flow(X_t, y_t, batch_size=BATCH_SIZE),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_FROZEN,
    validation_data=(X_v, y_v),
    callbacks=[early_stop, reduce_lr]
)

# --- 7. PHASE 2 : Fine-tuning des dernières couches du backbone ---
print("\n=== PHASE 2 : Fine-tuning (dernières 30 couches dégelées) ===")

# On dégèle les 30 dernières couches du backbone
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Learning rate plus faible pour ne pas écraser les poids pré-entraînés
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop2 = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr2 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history2 = model.fit(
    datagen.flow(X_t, y_t, batch_size=BATCH_SIZE),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_FINETUNE,
    validation_data=(X_v, y_v),
    callbacks=[early_stop2, reduce_lr2]
)

# --- 8. PRÉDICTIONS ---
print("\nGénération des prédictions pour Kaggle...")
preds       = model.predict(X_kaggle, batch_size=BATCH_SIZE)
final_preds = np.argmax(preds, axis=1)

val_preds         = model.predict(X_v, batch_size=BATCH_SIZE)
val_preds_classes = np.argmax(val_preds, axis=1)
y_true            = np.argmax(y_v, axis=1)

# --- 9. SUBMISSION CSV ---
submission = pd.DataFrame({
    'id':     np.arange(9227, 9227 + len(final_preds)),
    'target': final_preds
})
submission.to_csv('my_submission_v2.csv', index=False)
print("Fichier 'my_submission_v2.csv' prêt !")

# --- 10. MÉTRIQUES & VISUALISATIONS ---
val_loss, val_acc = model.evaluate(X_v, y_v, verbose=0)
print(f"\n--- RÉSULTATS FINAUX ---")
print(f"Accuracy de validation : {val_acc * 100:.2f} %")
print(f"Loss de validation     : {val_loss:.4f}")

print("\n--- RAPPORT DE CLASSIFICATION ---")
print(classification_report(y_true, val_preds_classes,
                            target_names=['0 Pers', '1 Pers', '2 Pers', '3 Pers']))

print(model.summary())

# Fusion des historiques des deux phases
acc  = history1.history['accuracy']      + history2.history['accuracy']
vacc = history1.history['val_accuracy']  + history2.history['val_accuracy']
loss = history1.history['loss']          + history2.history['loss']
vloss= history1.history['val_loss']      + history2.history['val_loss']
sep  = len(history1.history['accuracy']) # Epoch de séparation Phase1/Phase2

# Graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
axes[0].plot(acc,  label='Train Accuracy')
axes[0].plot(vacc, label='Val Accuracy')
axes[0].axvline(x=sep, color='gray', linestyle='--', label='Début fine-tuning')
axes[0].set_title("Évolution de l'Accuracy")
axes[0].legend()

# Loss
axes[1].plot(loss,  label='Train Loss')
axes[1].plot(vloss, label='Val Loss')
axes[1].axvline(x=sep, color='gray', linestyle='--', label='Début fine-tuning')
axes[1].set_title("Évolution de la Loss")
axes[1].legend()

# Matrice de confusion
cm = confusion_matrix(y_true, val_preds_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
axes[2].set_title('Matrice de Confusion')
axes[2].set_xlabel('Prédiction')
axes[2].set_ylabel('Vrai Label')

plt.tight_layout()
plt.show()
