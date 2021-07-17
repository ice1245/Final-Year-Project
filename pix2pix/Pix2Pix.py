
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torchvision.utils
from torch import nn
from torchvision.utils import save_image

from fastmri.models import Unet
from torch.nn import functional as F


from fastmri.pl_modules.mri_module import MriModule
from pathlib import Path

from pix2pix.PatchGAN import PatchGAN


def display_progress(cond, fake, real, epoch, filename, slice_num):
    cond = cond.detach().cpu()
    fake = fake.detach().cpu()
    real = real.detach().cpu()
    filename = filename[0][:-3]

    Path(f"output/{filename}").mkdir(parents=True, exist_ok=True)

    save_image(fake, "output/" + filename + f"/generated_{slice_num}_{epoch}.png")
    save_image(real, "output/" + filename + f"/target_{slice_num}.png")
    # save_image(cond * 0.5 + 0.5, "output/"+filename + f"/input_{slice_num}.png")


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Pix2Pix(MriModule):
    """
    Pix2Pix training module.

    But I have Used baseline U-Nets for generator in Pix2pix GAN.


    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
            self,
            in_chans=1,
            out_chans=1,
            chans=32,
            num_pool_layers=4,
            drop_prob=0.0,
            lr=0.002,
            lr_step_size=40,
            lr_gamma=0.1,
            weight_decay=0.0,
            **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.gen = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )
        self.save_hyperparameters()

        self.display_step = 100

        self.patch_gan = PatchGAN(1 + 1)

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditioned_images)
        disc_logits = self.patch_gan(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = 200

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()

        fake_logits = self.patch_gan(fake_images, conditioned_images)

        real_logits = self.patch_gan(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def training_step(self, batch, batch_idx, optimizer_idx):
        condition, real, mean, std, fname, slice_num, max_value = batch
        loss = None

        if optimizer_idx == 0:
            loss = self._disc_step(real.unsqueeze(1), condition.unsqueeze(1))
            self.log('PatchGAN Loss', loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(real.unsqueeze(1), condition.unsqueeze(1))
            self.log('Generator Loss', loss)

        if slice_num > 18 and slice_num < 26:
            fake = self.gen(condition.unsqueeze(1)).detach()
            display_progress(condition, fake.squeeze(1), real, self.current_epoch, fname, slice_num)

        return loss

    # def on_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)
    #     sample_image = self(z)
    #     grid = torchvision.utils.make_grid(sample_image)
    #     self.logger.experiment.ad

    # def validation_step(self, batch, batch_idx):
    #     image, target, mean, std, fname, slice_num, max_value = batch
    #     output = self.gen(image.unsqueeze(1))
    #     mean = mean.unsqueeze(1).unsqueeze(2)
    #     std = std.unsqueeze(1).unsqueeze(2)

    #     return {
    #         "batch_idx": batch_idx,
    #         "fname": fname,
    #         "slice_num": slice_num,
    #         "max_value": max_value,
    #         "output": output * std + mean,
    #         "target": target * std + mean,
    #         "val_loss": F.l1_loss(output, target),
    #     }

    def test_step(self, batch, batch_idx):
        image, real, mean, std, fname, slice_num, _ = batch
        # if slice_num == 22:
        fname = fname[0][:-3]
        output = self.gen(image.unsqueeze(1)).detach()
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        # Path(f"Users/17-SE-31/Test/{fname}").mkdir(parents=True, exist_ok=True)

        # save_image(image, "Users/17-SE-31/Test/"+fname + f"/input_{slice_num}.png")

        # save_image(output, "Users/17-SE-31/Test/"+fname + f"/generated_{slice_num}.png")

        return {
            "fname": fname,
            "slice": slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        lr = 0.0002
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr)
        return disc_opt, gen_opt

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.002, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
