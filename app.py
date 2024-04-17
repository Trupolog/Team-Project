import io
import json
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
import os
from joblib import dump, load

app = Flask(__name__)

# Загрузка соответствия между классами и названиями видов
with open('plantnet300K_species_names.json', 'r') as f:
    species_names = json.load(f)

data_dir = ''

train_dir = os.path.join(data_dir, 'images_train')
val_dir = os.path.join(data_dir, 'images_val')
test_dir = os.path.join(data_dir, 'images_test')

true_names = load('true_names.joblib')

# Определение количества классов
num_classes = len(species_names)

# Загрузка обученной модели
model = models.resnet50(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('model_best_accuracy.pth', map_location=torch.device('mps')))
model.eval()

# Определение функции предобработки изображения
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Определение маршрута для обработки запросов
@app.route('/predict', methods=['POST'])
def predict():
    print("Получен запрос")
    if 'image' not in request.files:
        print("Изображение не найдено в запросе")
        return jsonify({'error': 'No image found in the request'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    print("Изображение получено")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        print("Изображение открыто")
    except IOError:
        print("Неверный формат изображения")
        return jsonify({'error': 'Invalid image format'}), 400

    image = preprocess_image(image)
    print("Изображение предобработано")

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = preds.item()
        print('Predicted class:', predicted_class)
        # predicted_species = species_names[str(predicted_class)]
        # predicted_class = image_datasets['test'].classes[preds[0]]
        class_id_str = true_names[predicted_class]
        real_name = species_names[class_id_str]
        print("Предсказание выполнено")

    print("Отправка ответа")
    return jsonify({'class': predicted_class, 'real_name': real_name})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
#%%