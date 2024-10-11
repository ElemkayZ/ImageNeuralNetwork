import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

#data processing transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale(going to break without it)
    transforms.ToTensor(),
    transforms.Normalize((0.2510), (0.2898))  # Mean and std deviation for normalization use calcMeanAndStdForDataSet.py to calc for given dataset
])
#dataset Import
train_dataset = datasets.ImageFolder(root='dataSet Path', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#neural network definition
class Neuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linearReluStack = nn.Sequential(
            nn.Linear(590*445, 512),#neuron input size(img size) and number of neurons
            nn.ReLU(),
            nn.Linear(512, 512),#neuron input and neuron number
            nn.ReLU(),
            nn.Linear(512,9)#instead of 9 put number of classes in dataset
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits

#using gpu for processing
device = "cuda" if torch.cuda.is_available() else "cpu" 
model = Neuron().to(device)

# Recreate the model architecture
model = Neuron()  
model.load_state_dict(torch.load("trained_model.pth",weights_only=True))
model.to(device)
model.eval()  # Set the model to evaluation mode needed in some datasets depending on used training model

#single image prediction
def predictSingleImage(image_path, model, transform, device):
    # Open image and apply transforms
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension 
    image_tensor = image_tensor.to(device)  #convert into a tensor

    # Make prediction
    with torch.no_grad():  # No need to calculate gradients
        logits = model(image_tensor)  # Forward pass
        pred_probab = nn.Softmax(dim=1)(logits)  # Convert logits to probabilities
        y_pred = pred_probab.argmax(1)  # Get the predicted class
    return y_pred.item()  # Return the predicted label


while(True):
    image_path = input("give a photo path: ")  # Provide the path to the image you want to predict on
    predicted_label = predictSingleImage(image_path, model, transform, device)

    class_name = train_dataset.classes[predicted_label]
    print(f"Predicted class: {predicted_label} ({class_name})")