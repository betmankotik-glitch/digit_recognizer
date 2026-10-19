import os
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist as tf_mnist

# ГЛОБАЛЬНІ ЗМІННІ
MODEL_PATH = "model.h5"
mode = "train"  # train, predict

# СТВОРЕННЯ АБО ЗАВАНТАЖЕННЯ МОДЕЛІ
if os.path.exists(MODEL_PATH):
    mode = "predict"
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    mode = "train"
    print("Creating new model...")
    model = Sequential([
        Input(shape=(784,)),
        Dense(128, activation="relu"),
        Dense(256, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def preprocess_image(img):
    """Конвертація зображення PIL у нормалізований масив numpy розміром (1, 784)"""
    # Інвертуємо кольори, якщо потрібно ( MNIST має білі цифри на чорному тлі або навпаки,
    # залежно від того, як ви малюєте. Тут налаштовано під чорне на білому).
    arr = np.array(img).astype("float32") / 255.0
    # Згладжуємо у вектор з 784 елементів
    arr = arr.reshape(1, 784)
    return arr


def predict(img):
    """Розпізнавання намальованої цифри"""
    global model
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    print(f"Predicted digit: {predicted_digit}")
    return int(predicted_digit)


# Будемо збирати дані для донавчання в пам'яті
train_images_buffer = []
train_labels_buffer = []


def train(img, label_value, train_count):
    """Додавання зображення у буфер та навчання, коли набереться достатньо даних"""
    global train_images_buffer, train_labels_buffer, model

    processed_img = np.array(img).astype("float32") / 255.0
    train_images_buffer.append(processed_img.reshape(784))
    train_labels_buffer.append(label_value)

    print(f"Collected {len(train_images_buffer)}/{train_count} samples.")

    if len(train_images_buffer) >= train_count:
        X_train = np.array(train_images_buffer)
        y_train = np.array(train_labels_buffer)

        print("Training model with user data...")
        model.fit(X_train, y_train, epochs=5, batch_size=len(X_train), verbose=1)

        # Збереження моделі
        model.save(MODEL_PATH)
        print("Model saved successfully!")

        # Очищення буфера
        train_images_buffer.clear()
        train_labels_buffer.clear()
        return True

    return False


def mnist():
    """Навчання моделі на стандартному датасеті MNIST"""
    global model, mode
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf_mnist.load_data()

    # Нормалізація та випрямлення
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    # ПЕРЕКОМПІЛЯЦІЯ МОДЕЛІ (вирішує проблему з Unknown variable)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training on MNIST dataset (this may take a minute)...")
    model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))

    model.save(MODEL_PATH)
    mode = "predict"
    print("MNIST training finished and model saved!")
