import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# IMPORTA IL MODELLO GIUSTO
from models.model import MyModel  # <-- controlla questo!

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = ImageFolder(
    root='/dataset/tiny-imagenet-200/train',  # oppure val se sistemato
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


def evaluate():
    model = MyModel()
    model.load_state_dict(torch.load("checkpoints/model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    evaluate()