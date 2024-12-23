import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#data processing transform
transform = transforms.Compose([
    transforms.Resize((128,128)),#depends on training model
    #transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder(root='TrainDatasetPath', transform=transform)

# Calculate mean and std for your dataset
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    #mean = 0.0
    #std = 0.0

    mean = torch.zeros(3)  # Initialize mean for 3 channels (RGB)
    std = torch.zeros(3)   # Initialize std for 3 channels (RGB)

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

mean, std = calculate_mean_std(train_dataset)
print("Mean:", mean)
print("Std:", std)  
