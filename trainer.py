import torch
import torch.nn.functional as F
import torchvision
from warnings import warn
import os
import itertools
from losses import *


class Trainer(object):
    def __init__(self, encoder, decoder, discriminator, optimGen, optimDis, adversarial='minimax', device=torch.device('cuda:0'), batch_size=32,
                 epochs=20, samples=8, checkpoints='./aae', recon='./recon/', test_noise=None, **kwargs):
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.discriminator = discriminator.to(device)
        self.adversarial = adversarial
        self.nrow = 8 if 'nrow' not in kwargs else kwargs['nrow']
        if 'optim_gen_opt' in kwargs:
            self.optimGen = optimGen(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), **kwargs['optim_gen_opt'])
        else:
            self.optimGen = optimGen(itertools.chain(self.encoder.parameters(), self.decoder.parameters()))

        if 'optim_dis_opt' in kwargs:
            self.optimDis = optimDis(self.discriminator.parameters(), **kwargs['optim_dis_opt'])
        else:
            self.optimDis = optimDis(self.discriminator.parameters())

        self.batch_size = batch_size
        self.epochs = epochs
        self.start_epoch = 0
        self.checkpoints = checkpoints
        self.recon = recon
        self.samples = samples
        self.autoencoder_losses = []
        self.generator_losses = []
        self.discriminator_losses = []
        self.test_noise = torch.randn(self.samples, self.encoder.z_dim, device=self.device) if test_noise is None else test_noise

    def save_model(self, epoch):
        save_path = self.checkpoints + '.model'
        model = {
            'epoch': epoch + 1,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'adversarial': self.adversarial,
            'optimGen': self.optimGen.state_dict(),
            'optimDis': self.optimDis.state_dict(),
            'autoencoder_losses': self.autoencoder_losses,
            'generator_losses': self.generator_losses,
            'discriminator_losses': self.discriminator_losses,
        }
        torch.save(model, save_path)

    def load_model(self, load_path=''):
        load_path = self.checkpoints + '.model'
        print("Loading Model From '{}'".format(load_path))
        try:
            check = torch.load(load_path)
            self.start_epoch = check['epoch']
            self.encoder.load_state_dict(check['encoder'])
            self.decoder.load_state_dict(check['decoder'])
            self.discriminator.load_state_dict(check['discriminator'])
            self.optimGen.load_state_dict(check['optimGen'])
            self.optimDis.load_state_dict(check['optimDis'])
            self.optimDis.load_state_dict(check['optimDis'])
            self.autoencoder_losses = check['autoencoder_losses']
            self.generator_losses = check['generator_losses']
            self.discriminator_losses = check['discriminator_losses']
        except:
            warn('Model could not be loaded from {}. Training from Scratch'.format(load_path))
            self.start_epoch = 0
            self.encoder_losses = []
            self.decoder_losses = []
            self.discriminator_losses = []

    def sample_images(self, epoch):
        os.makedirs(os.path.dirname('{}/epoch{}.png'.format(self.recon, epoch + 1)), exist_ok=True)
        save_path = '{}/epoch{}.png'.format(self.recon, epoch + 1)
        print('Generating and Saving Images to {}'.format(save_path))
        with torch.no_grad():
            images = self.decoder(self.test_noise)
            img = torchvision.utils.make_grid(images)
            torchvision.utils.save_image(img, save_path, nrow=self.nrow)

    def train(self, data_loader, **kwargs):
        for epoch in range(self.start_epoch, self.epochs):
            print('Running Epoch {}'.format(epoch + 1))
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()
            running_autoencoder_losses = 0.0
            running_generator_losses = 0.0
            running_discriminator_losses = 0.0
            for i, data in enumerate(data_loader, 1):
                if type(data) is tuple or type(data) is list:
                    inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)
                else:
                    inputs = data[0].to(self.device)

                self.optimGen.zero_grad()
                # --------GENERATOR TRAINING - RECONSTRUCTION + REGULARIZATION------------------
                z, _, _ = self.encoder(inputs)
                recon, _, _ = self.decoder(z)
                recon_loss = F.mse_loss(inputs, recon)
                if self.adversarial == 'minimax':
                    gen_loss = minimax_generator_loss(self.discriminator(z))
                elif self.adversarial == 'least_squares':
                    gen_loss = least_squares_generator_loss(self.discriminator(z))
                elif self.adversarial == 'wasserstein':
                    gen_loss = wasserstein_generator_loss(self.discriminator(z))
                else:
                    raise Exception('Invalid adversarial loss')
                total_gen_loss = recon_loss + gen_loss
                total_gen_loss.backward()
                self.optimGen.step()
                # --------REGULARIZATION PHASE - DISCRIMINATOR TRAINING-------------
                self.optimDis.zero_grad()
                real_sample = torch.randn_like(z)
                grad_penalty = None
                if self.adversarial == 'minimax':
                    dis_loss = minimax_discriminator_loss(self.discriminator(real_sample), self.discriminator(z.detach()))
                elif self.adversarial == 'least_squares':
                    dis_loss = least_squares_discriminator_loss(self.discriminator(real_sample), self.discriminator(z.detach()))
                elif self.adversarial == 'wasserstein':
                    dis_loss = wasserstein_discriminator_loss(self.discriminator(real_sample), self.discriminator(z.detach()))
                    eps = torch.randn(1).item()
                    interpolate = eps * real_sample + (1 - eps) * z
                    d_interpolate = self.discriminator(interpolate)
                    grad_penalty = wasserstein_gradient_penalty(interpolate, d_interpolate)
                    if 'wasserstein_lambda' in kwargs:
                        grad_penalty *= kwargs['wasserstein_lambda']
                else:
                    raise Exception('Invalid adversarial loss')
                total_dis_loss = dis_loss if grad_penalty is None else dis_loss + grad_penalty
                total_dis_loss.backward()
                self.optimDis.step()

                running_autoencoder_losses += recon_loss.item()
                running_generator_losses += gen_loss.item()
                running_discriminator_losses += dis_loss.item()
                if grad_penalty is not None:
                    running_discriminator_losses += grad_penalty.item()
            self.save_model(epoch)
            self.encoder.eval()
            self.decoder.eval()
            self.discriminator.eval()
            self.autoencoder_losses.append(running_autoencoder_losses / i)
            self.generator_losses.append(running_generator_losses / i)
            self.discriminator_losses.append(running_discriminator_losses / i)
            self.sample_images(epoch)
            print('Epoch {} : Autoencoder Loss : {} Generator Adversarial Loss : {} Discriminator Adversarial Loss : {}'.format(epoch + 1, running_autoencoder_losses / i, running_generator_losses / i, running_discriminator_losses / i))
