from torch.nn import Module
from torch import nn
from torchsummary import summary


class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 1), padding=(2, 0))
        self.relu1 = nn.ReLU()
        self.max_pooling1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), padding=(1, 0))
        self.max_pooling2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), padding=(1, 0))
        self.max_pooling3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4096 * 13, 8)

    def forward(self, input_tensor1, input_tensor2):
        output_tensor1 = self.max_pooling1(self.relu1(self.conv1(input_tensor1)))
        output_tensor1 = self.max_pooling2(self.relu2(self.conv2(output_tensor1)))
        output_tensor1 = self.max_pooling3(self.relu3(self.conv3(output_tensor1)))
        output_tensor1 = self.flatten(output_tensor1)
        output_tensor1 = self.fc(output_tensor1)

        output_tensor2 = self.max_pooling1(self.relu1(self.conv1(input_tensor2)))
        output_tensor2 = self.max_pooling2(self.relu2(self.conv2(output_tensor2)))
        output_tensor2 = self.max_pooling3(self.relu3(self.conv3(output_tensor2)))
        output_tensor2 = self.flatten(output_tensor2)
        output_tensor2 = self.fc(output_tensor2)

        return output_tensor1, output_tensor2


if __name__ == '__main__':
    model = CustomModel()
    summary(model, [(1, 2048, 13), (1, 2048, 13)], batch_size=2)
    pass
