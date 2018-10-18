import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from losses import *
from model import *
from trainer import *

dset = torchvision.datasets.MNIST(root='./dataset/mnist/', download=True, train=True, transform=T.Compose([T.Pad(2), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
loader = data.DataLoader(dset, batch_size=32, shuffle=True, num_workers=4)
enc = Encoder(input_channels=1)
dec = Decoder(target_channels=1)
dis = Discriminator()
trainer = Trainer(enc, dec, dis, optim.Adam, optim.SGD, epochs=50, samples=64, adversarial='least_squares', optim_gen_opt = {'lr': 0.0002, 'betas': (0.5, 0.999)}, optim_dis_opt = {'lr': 0.01})
trainer.train(loader)
