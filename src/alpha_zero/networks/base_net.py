import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class BaseNet(nn.Module):
    def __init__(self, input_channels: int, board_size: int, action_size: int, num_res_blocks: int = 10, hidden_channels: int = 256, value_hidden: int = 128, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_channels)
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_channels) for _ in range(num_res_blocks)])

        # Policy head: conv to reduce, then linear to action_size
        self.policy_conv = nn.Conv2d(hidden_channels, 64, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(64)
        policy_flat_size = 64 * board_size * board_size
        self.policy_fc = nn.Linear(policy_flat_size, action_size)

        # Value head: conv to 1 channel, then linear
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        value_flat_size = 1 * board_size * board_size
        self.value_fc1 = nn.Linear(value_flat_size, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple:
        x = x.to(self.device)
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        # Policy
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)

        # Value
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value