#Trainig using ensemble methods
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping

# ================= PATHS ===================
aggressive_path = "/content/drive/MyDrive/Audio_training/aggressive"
nonagg_path = "/content/drive/MyDrive/Audio_training/non_aggressive"

# ================= LOAD DATA ===================
X = []
y = []

for file in os.listdir(aggressive_path):
    if file.endswith(".npy"):
        X.append(np.load(os.path.join(aggressive_path, file)))
        y.append(1)

for file in os.listdir(nonagg_path):
    if file.endswith(".npy"):
        X.append(np.load(os.path.join(nonagg_path, file)))
        y.append(0)

X = np.array(X)
y = np.array(y)

print("Data Loaded:", X.shape, " Labels:", y.shape)

# Reshape for CNN
X = np.expand_dims(X, axis=-1)  # shape -> (N,64,157,1)

# ================= SPLIT ===================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ================= CNN MODEL ===================
input_layer = Input(shape=(64,157,1))

x = layers.Conv2D(32, (3,3), activation='relu')(input_layer)
x = layers.MaxPool2D((2,2))(x)

x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPool2D((2,2))(x)

x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.Flatten()(x)

x = layers.Dense(128, activation='relu', name="feature_output")(x)
output_layer = layers.Dense(1, activation='sigmoid')(x)

cnn_model = models.Model(inputs=input_layer, outputs=output_layer)

cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

# ================= TRAIN CNN ===================
callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = cnn_model.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=16,
    epochs=25,
    callbacks=[callback]
)

# ================= EVALUATE ===================
test_pred = cnn_model.predict(X_test)
test_pred = (test_pred > 0.5).astype(int)

print("CNN Accuracy :", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# ================= EXTRACT EMBEDDINGS ===================
feature_model = models.Model(inputs=cnn_model.input,
                             outputs=cnn_model.get_layer("feature_output").output)

train_features = feature_model.predict(X_train)
test_features = feature_model.predict(X_test)

print("Embedding Shape:", train_features.shape)

# ================= ENSEMBLE LEARNING ===================
rf = RandomForestClassifier(n_estimators=200)
svm = SVC(probability=True)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05)

# Soft Voting Ensemble
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("svm", svm), ("xgb", xgb)],
    voting="soft"
)

ensemble.fit(train_features, y_train)
ens_pred = ensemble.predict(test_features)

print("Ensemble Accuracy:", accuracy_score(y_test, ens_pred))
print(classification_report(y_test, ens_pred))

# ================= SAVE MODELS ===================
cnn_model.save("/content/drive/MyDrive/aggression_cnn_model.h5")
import pickle
pickle.dump(ensemble, open("/content/drive/MyDrive/audio_model/ensemble_classifier.pkl", "wb"))

print("🎉 Training Complete & Models Saved Successfully!")
