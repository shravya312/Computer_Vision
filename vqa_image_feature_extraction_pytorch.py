# Import necessary libraries
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained ResNet50 model
# Use the latest weights if available
weights = models.ResNet50_Weights.DEFAULT
resnet50 = models.resnet50(weights=weights)

# Remove the classification layer (the last fully connected layer)
# We want the features before the classification
modules = list(resnet50.children())[:-1]
resnet50_features = torch.nn.Sequential(*modules)

# Set the model to evaluation mode
resnet50_features.eval()

# Get the image preprocessing transform
preprocess = weights.transforms()

# Function to extract features from an image
def extract_image_features_pytorch(img_path):
    # Load the image
    img = Image.open(img_path).convert('RGB')
    
    # Apply preprocessing transforms
    img_tensor = preprocess(img)
    
    # Add a batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    # Move the tensor to the appropriate device (CPU or GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    resnet50_features.to(device)
    
    # Get the features
    with torch.no_grad(): # Disable gradient calculation for inference
        features = resnet50_features(img_tensor)
    
    # Squeeze the batch dimension back (output shape will be [num_features])
    features = features.squeeze(0)
    
    return features

# Example usage (replace 'path/to/your/image.jpg' with an actual image file path)
# try:
#     image_features = extract_image_features_pytorch('path/to/your/image.jpg')
#     print("Extracted image features shape (PyTorch):", image_features.shape)
# except FileNotFoundError:
#     print("Error: Image file not found.")
# except Exception as e:
#     print(f"Error extracting features: {e}")

# ... existing code ...

# Note:
# This file only covers the image feature extraction part of a VQA project using PyTorch.
# A full VQA system also requires:
# 1. Text processing for the question.
# 2. Fusing image and text features.
# 3. Generating or classifying the answer.
# You would typically train a separate model or combine models to handle these steps.
# Consider exploring resources on VQA datasets (like VQA v2.0) and architectures (like deep learning models combining CNNs and LSTMs/Transformers) for a complete implementation. 