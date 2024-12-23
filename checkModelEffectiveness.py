import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ustawienia parametrów
batchSize = 64
imgX = 256
imgY = 256


# Data processing transform
transform = transforms.Compose([
    transforms.Resize((imgX, imgY)),
    transforms.RandomRotation(degrees=(0, 30)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3230, 0.3423, 0.3751], std=[0.2569, 0.2207, 0.2179])
])

# Dataset import
test_dataset = datasets.ImageFolder(root='testFolderPath', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
print("Number of classes in testing dataset:", len(test_dataset.classes))
print("Class to index mapping (test):", test_dataset.class_to_idx)

# Neural network definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        def compute_output_size(input_size, layers):
            H, W = input_size
            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    H = (H + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                    W = (W + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
                elif isinstance(layer, nn.MaxPool2d):
                    H = H // layer.kernel_size
                    W = W // layer.kernel_size
            return H, W

        H, W = compute_output_size((imgX, imgY), self.conv_layers)
        neurons = 128
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(neurons * H * W, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class Neuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        neuronNumber = 1024
        self.linearReluStack = nn.Sequential(
            nn.Linear(imgX*imgY*3, neuronNumber),#neuron input size(img size) and number of neurons
            nn.ReLU(),
            nn.Linear(neuronNumber, neuronNumber),#neuron input and neuron number
            nn.ReLU(),
            nn.Linear(neuronNumber, neuronNumber),#neuron input and neuron number
            nn.ReLU(),
            nn.Linear(neuronNumber, len(test_dataset.classes))#neuron transform into dataset
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits

# Check if GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Instantiate and move the model to the GPU if available
model = CNN(num_classes=len(test_dataset.classes)).to(device)
#model = Neuron()  
model.load_state_dict(torch.load('trainedModel.pth Path',weights_only=True,map_location=torch.device('cpu')))
model.to(device)
print("Model moved to", device)
model.eval()  # Set the model to evaluation mode needed in some datasets depending on used training model
# Evaluate model

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Call evaluation function
evaluate_model(model, test_loader, device)
