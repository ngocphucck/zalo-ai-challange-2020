import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch.optim import Adam
import torch.nn.functional as F


from voice_verification.src.models.vggvox import VGGVox
from voice_verification.src.losses.contrastive_loss import ContrastiveLoss
from datasets import VoiceContrastiveDataset
import config


def get_loader():
    train_dataset = VoiceContrastiveDataset(data_path='')
    test_dataset = VoiceContrastiveDataset(data_path='')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, test_loader


def train():
    train_loader, test_loader = get_loader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VGGVox(n_classes=400).to(device)
    model.fc8 = nn.Linear(1024, 8)

    criterion = ContrastiveLoss().to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)

    train_epoch_iterator = tqdm(train_loader,
                                desc="Training (Step X) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True)

    best_accuracy = 0

    for epoch in range(config.n_epochs):
        model.train()
        train_batch_losses = []
        print(f'Epoch {epoch + 1}: ')

        for input_tensor1, input_tensor2, labels in train_epoch_iterator:

            input_tensor1 = input_tensor1.to(device)
            input_tensor2 = input_tensor2.to(device)
            labels = labels.to(device)

            output_tensor1, output_tensor2 = model(input_tensor1), model(input_tensor2)
            loss = criterion(output_tensor1, output_tensor2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_iterator.set_description(f"loss: {loss.item()}")
            train_batch_losses.append(loss.item())

        print('Train loss: ', sum(train_batch_losses) / len(train_batch_losses))
        n_true = 0
        n_sample = 0
        model.eval()
        for input_tensor1, input_tensor2, labels in test_loader:
            input_tensor1 = input_tensor1.to(device)
            input_tensor2 = input_tensor2.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                output_tensor1, output_tensor2 = model(input_tensor1), model(input_tensor2)
                loss = criterion(output_tensor1, output_tensor2, labels)
                print(loss.item())
                euclidean_distance = F.pairwise_distance(output_tensor1, output_tensor2, keepdim=True)
                predicts = euclidean_distance
                predicts = predicts < 0.5
                predicts = 1 - predicts.view(-1).int()

            n_true += sum(predicts == labels)
            n_sample += predicts.shape[0]

        accuracy = n_true / n_sample
        print("Validation accuracy: ", accuracy)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'checkpoints/model.pth')


if __name__ == '__main__':
    train()
    pass
