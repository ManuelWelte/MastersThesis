
import torch.nn.functional as F

import torch


class Net(torch.nn.Module):

    def __init__(self):
        
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3)
        self.avgpool1 = torch.nn.AvgPool2d(2)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5)
        self.avgpool2 = torch.nn.AvgPool2d(2)

        self.fc1 = torch.nn.Linear(256, 200)
        self.fc2 = torch.nn.Linear(200, 10)
    
    def forward(self, X):
        
        out = self.avgpool2(F.relu(self.conv2(self.avgpool1(F.relu(self.conv1(X))))))

        out = torch.flatten(out, 1, -1)

        out = F.relu(self.fc1(out))

        return self.fc2(out)
    
    
    def get_features(self, X):
        out = self.avgpool2(F.relu(self.conv2(self.avgpool1(F.relu(self.conv1(X))))))

        out = torch.flatten(out, 1, -1)

        out = F.relu(self.fc1(out))
        return out