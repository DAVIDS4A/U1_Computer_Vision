import torch
from torch import nn
from preprocess import create_dataloaders, split_data, get_transforms
from get_data import PcPartsData
from clearml import Task

# Initialize ClearML Task
task = Task.init(project_name="PC Parts Classification", task_name="Training")

# Construct Convolution Neural Net
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = self._conv_block(3, 64)
        self.conv_block2 = self._conv_block(64, 128)
        self.conv_block3 = self._conv_block(128, 256)
        self.conv_block4 = self._conv_block(256, 512)
        self.conv_block5 = self._conv_block(512, 512)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 14)
        self.dropout = nn.Dropout(0.5)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Train Function Definition
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(torch.float32).to(device), y.to(torch.long).to(device)

        if torch.any(torch.isnan(X)) or torch.any(torch.isinf(X)):
            print("NaN or Inf detected in input data.")
            continue

        pred = model(X)
        loss = loss_fn(pred, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf detected in loss.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch % 100 == 0:
            loss_value, current = loss.item(), batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

# Validaiton function definition
def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(torch.float32).to(device), y.to(torch.long).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    img_dir = "pc_parts"
    annotations_file = "annotations.csv"
    train_transform, val_transform = get_transforms()
    dataset = PcPartsData(annotations_file=annotations_file, img_dir=img_dir, transform=train_transform)
    train_data, validation_data, test_data = split_data(dataset)
    train_loader, validation_loader = create_dataloaders(train_data, validation_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvNet().to(device).float()

    # Set model parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        validate(validation_loader, model, loss_fn)
    print("Done!")

    # Save model
    torch.save(model.state_dict(), "model_weights.pth")

    # Finalize ClearML Task
    task.close()
