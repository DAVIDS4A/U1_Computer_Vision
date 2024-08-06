import torch
from torch.utils.data import DataLoader
from preprocess import split_data, get_transforms
from get_data import PcPartsData
from train import ImprovedNeuralNet
from clearml import Task

# ClearML task setup
try:
    task = Task.init(project_name="PC Parts Classification", task_name="Inference", reuse_last_task_id=False)
except Exception as e:
    print(f"Error initializing task: {e}")
    Task.current_task().close()
    task = Task.init(project_name="PC Parts Classification", task_name="Inference", reuse_last_task_id=False)


def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(torch.float32).to(device)
            pred = model(X)
            predictions.append(pred.argmax(1).cpu().numpy())
    return predictions

if __name__ == "__main__":
    img_dir = "pc_parts"
    annotations_file = "annotations.csv"
    _, val_transform = get_transforms()
    dataset = PcPartsData(annotations_file=annotations_file, img_dir=img_dir, transform=val_transform)
    _, _, test_data = split_data(dataset)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImprovedNeuralNet().to(device).float()

    # Load the trained model weights
    model.load_state_dict(torch.load("model_weights.pth",weights_only=True))
    model.eval()

    predictions = predict(model, test_loader)
    print(predictions)

    # Finalize ClearML Task
    task.close()
