import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO

device = 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)

model.load_state_dict(torch.load('weights'))
model.to(device)