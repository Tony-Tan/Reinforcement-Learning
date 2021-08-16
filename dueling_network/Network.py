import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_channel_size, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channel_size, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_1_stat_value = nn.Linear(3136, 512)
        self.fc_1_advantage = nn.Linear(3136, 512)
        self.fc_2_stat_value = nn.Linear(512, 1)
        self.fc_2_advantage = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 3136)
        x_state_value = self.fc_1_stat_value(x)
        x_state_value = F.relu(x_state_value)
        x_state_value = self.fc_2_stat_value(x_state_value)

        x_advantage = self.fc_1_advantage(x)
        x_advantage = F.relu(x_advantage)
        x_advantage = self.fc_2_advantage(x_advantage)
        output = x_state_value + x_advantage - x_advantage.mean()
        return output
