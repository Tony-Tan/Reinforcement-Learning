import torch
import torch.nn as nn
import numpy as np


class Critic(torch.nn.Module):
    def __init__(self, input_state_size: int, input_action_size: int):
        super(Critic, self).__init__()
        self.linear_mlp_stack = nn.Sequential(
            nn.Linear(input_state_size+input_action_size, 1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        mu = self.linear_mlp_stack(x)
        return mu


def data_generator():
    x_1 = np.random.uniform(-1, 1, 1024).reshape([-1, 1])
    x_2 = np.random.uniform(-1, 1, 1024).reshape([-1, 1])
    output_np = 0*x_1 + 0 * x_2 + 1
    return np.concatenate((x_1, x_2), axis=1), output_np


if __name__ == '__main__':
    net = Critic(1, 1)
    loss = torch.nn.MSELoss()

    for i in range(1):
        data_in, data_out = data_generator()
        x = torch.tensor(data_in, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(data_out, dtype=torch.float32)

        output = net(x)
        loss_value = torch.sqrt(loss(y, output))
        gradient = torch.autograd.grad(outputs=output[0], inputs=x, retain_graph=True)
        print(gradient)
        gradient = torch.autograd.grad(outputs=output[1], inputs=x, retain_graph=True)
        print(gradient)






