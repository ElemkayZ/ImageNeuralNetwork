import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import datetime

batchSize = 64
imgX = 400
imgY = 400
day = datetime.datetime.now().strftime("%H-%M-%S")
CNNName = f"trainedModel-{imgX}x{imgY}-batch{batchSize}-CNN-{day}-5CNN3FC-FishSmallOpt.pth"
print(CNNName)

# Data processing transform
transform = transforms.Compose([
    transforms.Resize((imgX, imgY)),  # Resize to smaller dimensions suitable for CNN
    transforms.RandomRotation(degrees=(0, 30)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3230, 0.3423, 0.3751], std=[0.2569, 0.2207, 0.2179]) #fishDatasetSmallOpt
])

# Dataset import
train_dataset = datasets.ImageFolder(root='trainFolderPath', transform=transform)
test_dataset = datasets.ImageFolder(root='testFolderPath', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
print("Number of classes in training dataset:", len(train_dataset.classes))
print("Number of classes in testing dataset:", len(test_dataset.classes))
print("Class to index mapping (train):", train_dataset.class_to_idx)
print("Class to index mapping (test):", test_dataset.class_to_idx)

# Neural network definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 3 input channels (RGB), 32 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dimensions by half

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 512 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # 512 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
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
        neurons = 1024
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(1024 * 9 * 6, 1024),
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

# Check if GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Instantiate and move the model to the GPU if available
model = CNN(num_classes=len(train_dataset.classes)).to(device)
print("Model moved to", device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# Training loop
def train(model, train_loader, loss_fn, optimizer, device):
    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)  # Forward pass
        loss = loss_fn(pred, y)  # Compute loss
        
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        if batch % 16 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

# Test loop
def test(model, test_loader, loss_fn, device):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

# Train and evaluate the model
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(model, train_loader, loss_fn, optimizer, device)
    valLoss = test(model, test_loader, loss_fn, device)
    scheduler.step(valLoss)
    if valLoss < 0.15:
        break
    print(f"Learning Rate: {scheduler.get_last_lr()}")

# Save the trained model
torch.save(model.state_dict(), CNNName)
print("Training complete and model saved!")
