import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras

def predict_shape(image_path:str, img_size=64):
    # 1. Загрузка модели
    model_path = r'C:\Users\luba0\Desktop\my_geometry_app\models\my_model.keras'
    try:
        # Проверяем существование файла модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        print("Модель загружена успешно!")
    except Exception as e:
        print(f"Ошибка загрузки модели: {type(e).__name__}: {e}")
        return None  # Прекращаем выполнение при ошибке
    # 2. Предобработка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
       raise FileNotFoundError(f"Изображение не найдено или не может быть открыто: {image_path}")

    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized.astype('float32') / 255.0
    img_input = img_normalized.reshape(1, img_size, img_size, 1)  # Правильно: img_input

    # 3. Предсказание
    prediction = model.predict(img_input)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    # Названия классов (можно заменить на загрузку из модели, если доступно)
    classes = ['circle', 'square', 'triangle']
    predicted_class = str(classes[class_idx])

    result = {
        'class': predicted_class,
        'confidence': confidence,
        'image': img_resized,
        'prediction_probs': prediction[0]  # вероятности по всем классам
    }
    # 4. Вывод результата
    print(f"Предсказанный класс: {result['class']}")
    print(f"Уверенность: {result['confidence']:.4f}")
    print(f"Вероятности по классам: "
          f"круг={result['prediction_probs'][0]:.4f}, "
          f"квадрат={result['prediction_probs'][1]:.4f}, "
          f"треугольник={result['prediction_probs'][2]:.4f}")

    # 5. Визуализация
    plt.figure(figsize=(6, 6))
    plt.imshow(result['image'], cmap='gray')
    plt.title(f"Предсказание: {result['class']} "
              f"(уверенность: {result['confidence']:.2f})")
    plt.axis('off')
    plt.show()
    
    return predicted_class
    
    
    
