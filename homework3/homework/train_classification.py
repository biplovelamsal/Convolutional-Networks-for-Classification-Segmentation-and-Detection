import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms

from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data  # Provided by the assignment
from homework.metrics import AccuracyMetric


# ✅ Detect device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

# ✅ Simple normalization for validation
val_transform = transforms.Compose([
    transforms.ToTensor(),
])

# ✅ Load data (with explicit transform)
train_data = load_data("/content/classification_data/train")
val_data = load_data("/content/classification_data/val")

# If your dataset loader allows passing transforms, do:
# train_data = load_data("/content/homework3/classification_data/train", transform=train_transform)
# val_data = load_data("/content/homework3/classification_data/val", transform=val_transform)

# If not, it will already apply default transforms inside load_data()

train_loader = train_data
val_loader = val_data

# ✅ Initialize model, optimizer, and loss
model = Classifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
metric = AccuracyMetric()

# ✅ Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    # ✅ Validation
    model.eval()
    metric.reset()
    with torch.inference_mode():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model.predict(images)
            metric.add(preds, labels)

    val_acc = metric.compute()["accuracy"]
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

# ✅ Save trained model
save_model(model)
print("✅ Model saved successfully!")