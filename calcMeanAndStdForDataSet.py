from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#data processing transform
transform = transforms.Compose([
    transforms.Resize((474, 474)),#depends on training model
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder(root='DataSet Path', transform=transform)

# Calculate mean and std for your dataset
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

mean, std = calculate_mean_std(train_dataset)
print(mean, std)  # Use these values for normalization in data processing transform
