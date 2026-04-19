import torch
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0
from tqdm import tqdm

from Dataset import DigitDataset, _build_train_transform, _build_eval_transform, train_csv_path, trian_img_folder_path


def _build_model():
    model = efficientnet_b0()
    model.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 10)
    return model


if __name__ == '__main__':
    train_ds = DigitDataset(train_csv_path, trian_img_folder_path, transform=_build_train_transform())

    model = _build_model()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_split, val_split = random_split(train_ds, [train_size, val_size])

    train_dl = DataLoader(train_split, batch_size=64, num_workers=0)

    num_epochs = 10
    for epoch in range(num_epochs):
        loop = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in loop:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()

    val_dl = DataLoader(val_split, batch_size=64, num_workers=0)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dl:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
