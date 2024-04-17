import io
import json
from typing import Dict, Union
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os
from joblib import load

# Create a Flask application instance
app = Flask(__name__)

# Load the correspondence between class IDs and species names from a JSON file
# species_names: Dict[str, str] - Maps class IDs (as strings) to their corresponding species names
# The JSON file contains key-value pairs where the keys are class IDs and the values are species names
with open('plantnet300K_species_names.json', 'r') as f:
    species_names: Dict[str, str] = json.load(f)

# Set the base directory for the dataset
data_dir: str = ''

# Set the paths for the training, validation, and test directories
train_dir: str = os.path.join(data_dir, 'images_train')
val_dir: str = os.path.join(data_dir, 'images_val')
test_dir: str = os.path.join(data_dir, 'images_test')

# Load the true names mapping from a file using joblib
# true_names: Dict[int, str] - Maps the predicted class indices to their corresponding true class IDs
# The file contains a dictionary where the keys are predicted class indices and the values are true class IDs
true_names: Dict[int, str] = load('true_names.joblib')

# Determine the number of classes based on the length of the species names dictionary
num_classes: int = len(species_names)

# Load the pre-trained ResNet50 model with default weights
model = models.resnet50(weights='DEFAULT')

# Replace the last fully connected layer of the model to match the number of classes
# Get the number of input features for the last fully connected layer
num_ftrs: int = model.fc.in_features
# Create a new linear layer with the number of input features and output classes
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the trained model weights from a file
# The weights are loaded using torch.load and mapped to the 'mps' device (Apple Metal Performance Shaders)
model.load_state_dict(torch.load('model_best_accuracy.pth', map_location=torch.device('mps')))

# Set the model to evaluation mode (disables dropout and batch normalization)
model.eval()


# Define the image preprocessing function
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocesses the input image by resizing it to (256, 256) and converting it to a tensor.

    Args:
        image (Image.Image): The input image to be preprocessed.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    # Define a composition of image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to (256, 256)
        transforms.ToTensor(),  # Convert the image to a tensor
    ])
    # Apply the transformations to the input image
    image = transform(image)
    # Add a batch dimension to the image tensor (required for model input)
    image = image.unsqueeze(0)
    return image


# Define the route for handling prediction requests
@app.route('/predict', methods=['POST'])
def predict() -> Union[Dict[str, Union[int, str]], tuple]:
    """
    Handles the prediction request by receiving an image, preprocessing it, making a prediction using the trained model,
    and returning the predicted class and its corresponding species name.

    Returns:
        Union[Dict[str, Union[int, str]], tuple]: If the prediction is successful, returns a dictionary containing the predicted class and its species name.
                                                   If an error occurs, returns a tuple with an error message and an HTTP status code.
    """
    print("Request received")
    # Check if the request contains an 'image' file
    if 'image' not in request.files:
        print("No image found in the request")
        # Return an error response if no image is found
        return jsonify({'error': 'No image found in the request'}), 400

    # Get the image file from the request
    image_file = request.files['image']
    # Read the image file bytes
    image_bytes = image_file.read()
    print("Image received")

    try:
        # Open the image from the bytes using PIL (Python Imaging Library)
        image = Image.open(io.BytesIO(image_bytes))
        print("Image opened")
    except IOError:
        print("Invalid image format")
        # Return an error response if the image format is invalid
        return jsonify({'error': 'Invalid image format'}), 400

    # Preprocess the image using the preprocess_image function
    image = preprocess_image(image)
    print("Image preprocessed")

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Pass the image through the model to get the predicted output
        outputs = model(image)
        # Get the predicted class index by finding the maximum output value
        _, preds = torch.max(outputs, 1)
        predicted_class = preds.item()
        print('Predicted class:', predicted_class)
        # Get the true class ID corresponding to the predicted class index
        class_id_str = true_names[predicted_class]
        # Get the species name corresponding to the true class ID
        real_name = species_names[class_id_str]
        print("Prediction completed")

    print("Sending response")
    # Return the predicted class and its corresponding species name as a JSON response
    return jsonify({'class': predicted_class, 'real_name': real_name})


# Run the Flask application if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True, port=8000)