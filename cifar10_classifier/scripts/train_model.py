from torch.utils.data import DataLoader

def train_loop(dataloader: DataLoader, model, loss_fn, optimizer, print_every = 100):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % print_every == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f} batch: {batch:>5d}")