import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=1, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        with torch.no_grad():
            weights1 = torch.tensor([[[[0.2390, 0.1593], [0.5377, 0]]]])
            self.layer1[0].weight = nn.Parameter(weights1, requires_grad=False)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=1, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        with torch.no_grad():
            weights2 = torch.tensor([[[[-0.2390, -0.3585], [-0.5377, 0.2390]]]])
            self.layer2[0].weight = nn.Parameter(weights2, requires_grad=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def get_fmap(img_set):
    fmap_set = []
    model = CNN()
    for i in range(len(img_set)):
        x = torch.from_numpy(img_set[i]).float()
        temp = x.view([1, 1, 120, 160])
        out = model(temp)
        fmap_set.append(out)
    return fmap_set

