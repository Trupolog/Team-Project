# Team-Project
HSE TEAM PROJECT CV

Since GitHub has a 25MB file size limit, I'm providing the training data and other relevant files on Google Drive: [https://drive.google.com/drive/folders/1A0ieTFwd0OhFafQw72r-akKVJVnWOxfo?usp=sharing](https://drive.google.com/drive/folders/1A0ieTFwd0OhFafQw72r-akKVJVnWOxfo?usp=sharing).

### Project Overview

This project aims to classify plant species based on images, employing a deep learning approach. Given the vast diversity of plant species, this task presents a challenging yet fascinating problem in the field of computer vision and machine learning. Utilizing the PlantNet 300K dataset, this endeavor encompasses preprocessing steps, model training with fine-tuning, evaluation, and making predictions.

### Data Preparation and Preprocessing

The dataset, organized into training, validation, and testing subsets, undergoes several preprocessing steps to make it suitable for feeding into a convolutional neural network (CNN). These steps include:

- **Resizing**: All images are resized to a uniform dimension (256x256 pixels) to ensure consistency.
- **Augmentation**: To increase the dataset's diversity and robustness, various augmentations such as random rotations and horizontal flips are applied to the training data.
- **Normalization**: Image data is converted into PyTorch tensors and normalized to match the model's expected input format.

### Model Selection and Training

The ResNet-50 model, known for its depth and ability to learn rich feature representations, is chosen for this task. Pretrained on the ImageNet dataset, it offers a strong foundation for transfer learning. Key steps in model preparation include:

- **Parameter Freezing**: All layers, except the final fully connected layer, are frozen to retain the pretrained weights.
- **Output Layer Modification**: The final layer is modified to match the number of classes in the PlantNet dataset.
- **Optimizer and Loss Function**: The Adam optimizer and CrossEntropyLoss are selected for training the model.

Training involves iteratively passing the training data through the model, calculating the loss, and updating the model parameters. Validation data is used to evaluate the model's performance and implement early stopping to prevent overfitting.

### Evaluation and Prediction

Model performance is assessed using accuracy, calculated on both the validation and test datasets. This metric reflects the proportion of correctly predicted images against the total number of images.

Predictions are made by passing images through the trained model and obtaining the class with the highest predicted score. The predicted and actual class names are then displayed alongside the input image for visual verification.

### Challenges and Solutions

Key challenges in this project include managing the dataset's diversity and imbalance, tuning the model to adapt to the specific task, and ensuring the model generalizes well to unseen data. Solutions include leveraging data augmentation, fine-tuning a pretrained model, and employing early stopping based on validation performance.

### Conclusion

This project demonstrates the power of deep learning in classifying plant species from images. By leveraging a pretrained ResNet-50 model and fine-tuning it on the PlantNet 300K dataset, we achieve robust model performance. This approach underscores the effectiveness of transfer learning in applying deep neural networks to specific domains, even with the challenges of dataset diversity and imbalance.


