import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask import Flask, request, jsonify ,render_template

# Load the dataset
data = pd.read_csv("crop_data.csv")

# Encode categorical variables (e.g., soil type)
label_encoder = LabelEncoder()
data['soil_type'] = label_encoder.fit_transform(data['soil_type'])

# Feature selection
features = data.drop('crop', axis=1)  # Drop target variable
target = data['crop']

# Normalize numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# One-hot encoding target variable
target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)
y_test_encoded = target_encoder.transform(y_test)

# Ensure consistent one-hot encoding with all classes
num_classes = len(target_encoder.classes_)
y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

# Debug: Check class counts and consistency
print(f"Number of classes: {num_classes}")
print(f"Classes in target: {target_encoder.classes_}")

# Create the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Match output layer with number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_onehot, validation_split=0.1, epochs=50, batch_size=32)

# Debugging: Verify dimensions before evaluation
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test_onehot: {y_test_onehot.shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Number of classes in model output: {y_train_onehot.shape[1]}")

# Evaluate the model
try:
    accuracy = model.evaluate(X_test, y_test_onehot, verbose=1)
    print(f"Model Accuracy: {accuracy[1] * 100:.2f}%")
except Exception as e:
    print(f"Error during evaluation: {e}")

from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        # Assuming you have implemented the scaler and model
        input_features = scaler.transform([data['features']])
        predictions = model.predict(input_features)
        predicted_crop = target_encoder.inverse_transform([np.argmax(predictions)])
        return jsonify({'predicted_crop': predicted_crop[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/trade', methods=['POST'])
def trade_crop():
    try:
        trade_data = request.json
        return jsonify({'status': 'Trade successful!', 'trade_data': trade_data})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
