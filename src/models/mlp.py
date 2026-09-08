import torch
import torch.nn as nn

class MelanomaMLP(nn.Module):
    def __init__(self):
        super(MelanomaMLP, self).__init__()
        # Camada 1 
        self.fc1 = nn.Linear(in_features=224 * 224 * 3, out_features=512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.4)

        # Camada 2
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        # Camada 3 
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        # Camada de saída
        self.fc_out = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc_out(x)
        x = self.sigmoid(x)

        return x