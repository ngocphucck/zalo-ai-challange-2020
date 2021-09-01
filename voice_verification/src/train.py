import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from torch.optim import Adam


from src.models.custom_model import CustomModel
from src.losses.contrastive_loss import ContrastiveLoss
from datasets import VoicePairDataset
from utils import get_data
import config


def get_loader():
    data = get_data(data_folder=config.data_folder)
    train_data, test_data = train_test_split(data, test_size=config.test_size, shuffle=True, random_state=2021)
    train_dataset = VoicePairDataset(train_data, max_sequence_len=2048)
    test_dataset = VoicePairDataset(test_data, max_sequence_len=2048)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, test_loader


def train():
    train_loader, test_loader = get_loader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CustomModel().to(device)
    classifier = nn.Linear(16, 2).to(device)

    criterion = ContrastiveLoss().to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)

    clf_criterion = nn.CrossEntropyLoss().to(device)
    clf_optimizer = Adam(classifier.parameters(), lr=config.lr)

    train_epoch_iterator = tqdm(train_loader,
                                desc="Training (Step X) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True)

    best_accuracy = 0

    for epoch in range(config.n_epochs):
        train_batch_losses = []
        print(f'Epoch {epoch + 1}: ')

        for input_tensor1, input_tensor2, labels in train_epoch_iterator:
            model.train()

            input_tensor1 = input_tensor1.to(device)
            input_tensor2 = input_tensor2.to(device)
            labels = labels.to(device)

            output_tensor1, output_tensor2 = model(input_tensor1, input_tensor2)
            loss = criterion(output_tensor1, output_tensor2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            output_tensor1, output_tensor2 = model(input_tensor1, input_tensor2)
            output_tensor = torch.cat((output_tensor1, output_tensor2), dim=1)

            logits = classifier(output_tensor)
            loss = clf_criterion(logits, labels)
            train_batch_losses.append(loss.item())
            train_epoch_iterator.set_description(f"loss: {loss.item()}")

            clf_optimizer.zero_grad()
            loss.backward()
            clf_optimizer.step()

        n_true = 0
        n_sample = 0
        for input_tensor1, input_tensor2, labels in test_loader:
            input_tensor1 = input_tensor1.to(device)
            input_tensor2 = input_tensor2.to(device)
            labels = labels.to(device)

            output_tensor1, output_tensor2 = model(input_tensor1, input_tensor2)
            output_tensor = torch.cat((output_tensor1, output_tensor2), dim=1)
            logits = classifier(output_tensor)
            _, predicts = torch.max(logits, dim=1)

            n_true += sum(predicts == logits)
            n_sample += predicts.shape[0]

        accuracy = n_true / n_sample
        print("Validation accuracy: ", accuracy)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'checkpoints/model.pth')
            torch.save(classifier.state_dict(), 'checkpoints/clf.pth')


if __name__ == '__main__':
    pass
