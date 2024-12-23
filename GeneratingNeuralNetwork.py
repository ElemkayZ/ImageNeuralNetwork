import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import datetime
batchSize = 64
imgX = 256
imgY = 256
day = datetime.datetime.now().strftime("%H-%M-%S")
MLPName = f"trainedModel-{imgX}x{imgY}-batch{batchSize}-1024Neuronx3-MLP-{day}--FishSmallOpt.pth"
print(MLPName)

#data processing transform
transform = transforms.Compose([
    transforms.Resize((imgX,imgY)),
    #transforms.Grayscale(num_output_channels=1),  # Convert to grayscale(going to break without it)
    transforms.RandomRotation(degrees=(0, 30)),
    transforms.RandomHorizontalFlip(180),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3230, 0.3423, 0.3751], std=[0.2569, 0.2207, 0.2179]) #fishDatasetSmallOpt
])
#dataset Import
train_dataset = datasets.ImageFolder(root='trainFolderPath', transform=transform)
test_dataset = datasets.ImageFolder(root='testFolderPath', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
print("Number of classes in training dataset:", len(train_dataset.classes))
print("Number of classes in testing dataset:", len(test_dataset.classes))
print("Class to index mapping (train):", train_dataset.class_to_idx)
print("Class to index mapping (test):", test_dataset.class_to_idx)

#neural network definition
neuronNumber = 1024
class Neuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linearReluStack = nn.Sequential(
            nn.Linear(imgX*imgY*3, neuronNumber),#neuron input size(img size) and number of neurons
            nn.ReLU(),
            nn.Linear(neuronNumber, neuronNumber),#neuron input and neuron number
            nn.ReLU(),
            nn.Linear(neuronNumber, neuronNumber),#neuron input and neuron number
            nn.ReLU(),
            nn.Linear(neuronNumber, neuronNumber),#neuron input and neuron number
            nn.ReLU(),
            nn.Linear(neuronNumber, len(train_dataset.classes))#neuron transform into dataset
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits

#using gpu for processing
#Check if GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Verify GPU
if device == "cuda":
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


# Instantiate and move the model to the GPU if available
model = Neuron().to(device)
print("Model moved to", device)

# loss function
loss_fn = nn.CrossEntropyLoss()

#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#set up for training scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',  factor=0.1, patience=2)

#training loop
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
        
        #stats and training feedback
        if batch % 16 == 0:
            loss_val = loss.item()
            current = batch * len(X)

            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

#test loop
def test(model, test_loader, loss_fn, device):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    #prediction
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    #stats and testing feedback
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss
#Train and evaluate the model
epochs = 10 #number of dataset training reruns
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(model, train_loader, loss_fn, optimizer, device)
    valLoss = test(model, test_loader, loss_fn, device)
    scheduler.step(valLoss)  # Decrease learning rate, depends on the dateset if it's needed
    print(f"{scheduler.get_last_lr()}")

# Save the trained model
torch.save(model.state_dict(), MLPName)
print("Training complete and model saved!")
