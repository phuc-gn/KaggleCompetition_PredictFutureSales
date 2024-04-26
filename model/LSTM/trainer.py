import torch

from utils import get_device


def train(model, criteria, train_dataloader, optimiser, device, epoch, training_loss):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch + 1}")
    for idx, (input, output) in enumerate(train_dataloader):
        inputs, labels = input.to(device), output.to(device)

        optimiser.zero_grad()

        outputs = model(inputs)

        loss = criteria(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        if idx % 200 == 199:
            print(f" Batch {idx+1}/{len(train_dataloader)} - Loss: {running_loss / (idx+1)}")

    print(f"- Training Loss: {running_loss / len(train_dataloader)}")
    training_loss.append(running_loss / len(train_dataloader))


def val(model, criteria, val_dataloader, device, validation_loss):
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criteria(outputs, labels)

            running_loss += loss.item()

    print(f"- Validation Loss: {running_loss / len(val_dataloader)}")
    validation_loss.append(running_loss / len(val_dataloader))


def checkpoint(path, model):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location = device)
    model.load_state_dict(checkpoint)
    return model