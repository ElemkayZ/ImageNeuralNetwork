import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import date

#data processing transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale(going to break without it)
    transforms.ToTensor(),
    transforms.Normalize((0.2510), (0.2898))  # Mean and std deviation for normalization use calcMeanAndStdForDataSet.py to calc for given dataset
])
#dataset Import
train_dataset = datasets.ImageFolder(root='dataSet Path', transform=transform)
test_dataset = datasets.ImageFolder(root='dataSet Path', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
            nn.Linear(512, len(train_dataset.classes))#neuron transform into dataset
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits

#using gpu for processing
device = "cuda" if torch.cuda.is_available() else "cpu" 
model = Neuron().to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()

#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#set up for training scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
        if batch % 8 == 0:
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

#Train and evaluate the model
epochs = 10 #number of dataset training reruns
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(model, train_loader, loss_fn, optimizer, device)
    test(model, test_loader, loss_fn, device)
    scheduler.step()  # Decrease learning rate, depends on the dateset if it's needed

#saving trained model
day = date.today()
torch.save(model.state_dict(), "trainedModel-{day}.pth")
print("Training complete and model saved!")
