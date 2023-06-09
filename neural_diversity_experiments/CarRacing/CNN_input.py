import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.nn as nn
torch.set_num_threads(1)

class CNN(nn.Module):
    def __init__(self, input_channels=3):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=3, stride=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=2, bias=False)

        #self.conv3 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1, stride=1, bias=False)


    def forward(self, ob):
        with torch.no_grad():
            state = torch.as_tensor(ob.copy())
            state = state.float()

            x1 = torch.tanh(self.conv1(state))
            x1 = self.pool(x1)
            x2 = torch.tanh(self.conv2(x1))
            x2 = self.pool(x2)
            #x3 = torch.tanh(self.conv3(x2))
            x4 = x2.view(-1)

        return x4.detach().numpy()



