import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import Config

def cuda_(var):
    return var.cuda() if Config.use_gpu else var

class Net(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(state_dim, 20)
        self.linear2 = nn.Linear(20, num_actions)
        # self.W1 = nn.Parameter(torch.randn(state_dim, 20))
        # self.b1 = nn.Parameter(torch.randn(20))
        # self.W2 = nn.Parameter(torch.randn(20, num_actions))
        # self.b2 = nn.Parameter(torch.randn(num_actions))

        # self.myparameters = nn.ParameterList([nn.Parameter(self.W1), nn.Parameter(self.W2),
        #                                       nn.Parameter(self.b1), nn.Parameter(self.b2)])

    def forward(self, states, bit_vecs=None):
        h1 = torch.tanh(self.linear1(states))
        p = self.linear2(h1)
        import pdb
        # pdb.set_trace()
        p = F.log_softmax(p, dim=1)
        # if bit_vecs :
        #     if not isinstance(bit_vecs, torch.Tensor):
        #         bit_vecs = torch.tensor(bit_vecs, dtype=torch.float32, device=Config.device)
        #         bit_vecs.detach_()
        #     p = p * bit_vecs

        # h1 = F.tanh((torch.matmul(states, self.W1) + self.b1))
        # p = torch.matmul(h1, self.W2) + self.b2
        return p
