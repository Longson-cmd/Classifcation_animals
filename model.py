import torch.nn as nn
import torch

class Simple_model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = self.make_blocks(3, 8)
        self.conv2 = self.make_blocks(8, 16)
        self.conv3 = self.make_blocks(16, 32)
        self.conv4 = self.make_blocks(32, 64)
        self.conv5 = self.make_blocks(64, 64)
        
        self.fc1 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(64*7*7, 2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(1024, num_classes)

    def make_blocks(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) 

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    

if __name__ == '__main__':
    model = Simple_model()
    data = torch.rand(16, 3, 224, 224)
    outputs = model(data)
    # print(outputs)
    print(outputs.shape)