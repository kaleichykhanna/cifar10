import torch

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy
