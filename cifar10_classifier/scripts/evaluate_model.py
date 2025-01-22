import torch

def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_labels_no_dummies = [a.argmax() for a in all_labels]
    correct = sum(p == l for p, l in zip(all_preds, all_labels_no_dummies))
    total = len(all_labels) 
    accuracy = correct / total

    return accuracy

    