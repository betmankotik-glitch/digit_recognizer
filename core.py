import os
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam

# ГЛОБАЛЬНІ ЗМІННІ
MODEL_PATH = "model.h5"
mode = "train" # train, predict

# СТВОРЕННЯ АБО ЗАВАНТАЖЕННЯ МОДЕЛІ
if os.path.exists(MODEL_PATH):
    mode = "predict"
    print("Loading exists model...")
    model = load_model(MODEL_PATH)
    # Перекомпілюємо модель, щоб уникнути конфлікту оптимізатора
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
else:
    mode = "train"
    print("Creating new model...")
    model = Sequential([
        Dense(128, input_shape=(784,), activation="relu"),
        Dense(256, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

# Змінні для навчальних даних і підписів, зібраних через інтерфейс
train_images = []
train_labels = []


def train(img, number_value, train_count):
    global train_images, train_labels

    img_inverted = ImageOps.invert(img)
    img_array = np.array(img_inverted, dtype=np.float32) / 255.0
    img_flattened = img_array.reshape(784)

    train_images.append(img_flattened)
    train_labels.append(number_value)

    if len(train_images) >= train_count:
        x_train = np.array(train_images)
        y_train = np.array(train_labels)

        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)
        model.save(MODEL_PATH)

        train_images.clear()
        train_labels.clear()
        return True

    return False


def predict(img):
    img_inverted = ImageOps.invert(img)
    img_array = np.array(img_inverted, dtype=np.float32) / 255.0
    img_flattened = img_array.reshape(1, 784)

    prediction = model.predict(img_flattened)
    return int(np.argmax(prediction))


def mnist():
    global model, mode
    images = []
    labels = []

    train_dir = "mnist/train"
    if not os.path.exists(train_dir):
        print(f"Папка {train_dir} не знайдена!")
        return

    print("Завантаження даних MNIST для тренування...")
    for img_name in os.listdir(train_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(train_dir, img_name)
            try:
                label = int(img_name[0])
                img = Image.open(img_path).convert("L").resize((28, 28))
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array.reshape(784))
                labels.append(label)
            except Exception:
                pass

    if len(images) > 0:
        x_train = np.array(images)
        y_train = np.array(labels)

        print("Розпочато навчання моделі на датасеті MNIST...")
        model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
        model.save(MODEL_PATH)
        mode = "predict"
        print("Модель успішно навчена на MNIST та збережена у model.h5 ✅")
    else:
        print("Помилка: не знайдено зображень у папці mnist/train.")


def predict_mnist():
    test_dir = "mnist/test"
    if not os.path.exists(test_dir):
        print(f"Папка {test_dir} не знайдена!")
        return

    images = []
    labels = []

    print("Завантаження тестових даних MNIST...")
    for img_name in os.listdir(test_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, img_name)
            try:
                label = int(img_name[0])
                img = Image.open(img_path).convert("L").resize((28, 28))
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array.reshape(784))
                labels.append(label)
            except Exception:
                pass

    if len(images) > 0:
        x_test = np.array(images)
        y_test = np.array(labels)

        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"\n--- Результати тестування MNIST ---")
        print(f"Точність (Accuracy): {accuracy * 100:.2f}%")
        print(f"Втрати (Loss): {loss:.4f}")
    else:
        print("Помилка: не знайдено тестових зображень у папці mnist/test.")