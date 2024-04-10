import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch


class DQNAtari(nn.Module):
    def __init__(self, input_channel_size: int, output_size: int):
        super(DQNAtari, self).__init__()
        self.conv1 = nn.Conv2d(input_channel_size, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.fc_2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # self.bn1(x))
        x = self.conv2(x)
        x = F.relu(x)  # self.bn2(x))
        x = self.conv3(x)
        x = F.relu(x)  # self.bn3(x))
        x = x.view(-1, 3136)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x
# advanced network
# class DeepQNetwork(nn.Module):
#     def __init__(self, action_size, hidden_size):
#         super(DeepQNetwork, self).__init__()
#         self.conv_layer_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
#         self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv_layer_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.fc_layer = nn.Linear(7 * 7 * 64, hidden_size)
#         # V(s) value of the state
#         self.dueling_value = nn.Linear(hidden_size, 1)
#         # Q(s,a) Q values of the state-action combination
#         self.dueling_action = nn.Linear(hidden_size, action_size)
#
#     def forward(self, x):
#         x = F.relu(self.conv_layer_1(x))
#         x = F.relu(self.conv_layer_2(x))
#         x = F.relu(self.conv_layer_3(x))
#         x = F.relu(self.fc_layer(x.view(x.size(0), -1)))
#         # get advantage by subtracting dueling action mean from dueling action
#         # then add estimated state value
#         x = self.dueling_action(x) - self.dueling_action(x).mean(dim=1, keepdim=True) + self.dueling_value(x)
#         return x
