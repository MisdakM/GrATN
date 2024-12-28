
import torch
from utils import calculate_accuracy

def test_model(model, test_loader, criterion, device, weights_path):
    """Load model weights and evaluate on the test set."""
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy
