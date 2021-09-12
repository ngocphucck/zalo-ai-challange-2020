import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


from voice_verification.src.models.vggvox import VGGVox
from datasets import VoiceSingleDataset
import config


def get_loader():
    train_dataset = VoiceSingleDataset(data_path="")
    test_dataset = VoiceSingleDataset(data_path="")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, test_loader


def train():
    train_loader, test_loader = get_loader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VGGVox(n_classes=400).to(device)

    criterion = CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)

    train_epoch_iterator = tqdm(train_loader,
                                desc="Training (Step X) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True)

    best_accuracy = 0

    for epoch in range(config.n_epochs):
        train_batch_losses = []
        print(f'Epoch {epoch + 1}: ')

        for input_tensor, labels in train_epoch_iterator:
            model.train()

            input_tensor = input_tensor.to(device)
            labels = labels.to(device)

            output_tensor = model(input_tensor)
            loss = criterion(output_tensor, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_iterator.set_description(f"loss: {loss.item()}")
            train_batch_losses.append(loss.item())
        print('Train loss: ', sum(train_batch_losses) / len(train_batch_losses))

        n_true = 0
        n_sample = 0
        model.eval()
        for input_tensor, labels in test_loader:
            input_tensor = input_tensor.to(device)
            labels = labels.to(device)

            logits = model(input_tensor)
            _, predicts = torch.max(logits, dim=1)

            n_true += sum(predicts == logits)
            n_sample += predicts.shape[0]

        accuracy = n_true / n_sample
        print("Validation accuracy: ", accuracy)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'identification_checkpoints/model.pth')


if __name__ == '__main__':
    pass
