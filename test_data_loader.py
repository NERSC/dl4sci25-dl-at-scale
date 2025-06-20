import torch
from utils.data_loader import get_data_loader
import numpy as np
from utils.YParams import YParams
from networks.vit import ViT
import matplotlib.pyplot as plt
import os

os.makedirs('figs', exist_ok=True)

# on Perlmutter, run with:
# shifter --image=nersc/pytorch:24.08.01 -V /pscratch/sd/s/shas1693/data/dl-at-scale-training-data:/data python test_data_loader.py 

params = YParams('./config/ViT.yaml', 'short')
params.global_batch_size = 1
params.local_batch_size = 1
device = torch.device("cuda:0")
params.device = device

valid_dataloader, dataset_valid  = get_data_loader(params, params.valid_data_path, distributed=False, train=False)

model = ViT(params)
model = model.to(device)

with torch.no_grad():
  for i, data in enumerate(valid_dataloader, 0):
    if i >= 1:
        break
    print("Doing iteration {}".format(i))
    inp, tar = map(lambda x: x.to(device, dtype = torch.float), data)
    print("input shape = {}".format(inp.shape))
    print("target shape = {}".format(tar.shape))
    plt.rcParams["figure.figsize"] = (10,40)
    plt.figure()
    for ch in range(inp.shape[1]):
        plt.subplot(inp.shape[1],1, ch+1)
        plt.imshow(inp[0,ch,:,:].cpu(), cmap = 'RdBu')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
    plt.savefig("figs/minibatch_" + str(i) + ".pdf", dpi=1000)
    gen = model(inp)
    print("prediction shape = {}".format(gen.shape))

