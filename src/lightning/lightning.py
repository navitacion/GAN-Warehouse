import os, glob, random, time
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision.utils import make_grid
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.losses import KL_R_Loss, Wasserstein, GradientPaneltyLoss



# VAE - Lightning Module ---------------------------------------------------------------------------
class VAE_LightningSystem(pl.LightningModule):
    def __init__(self, net, cfg, experiment):
        super(VAE_LightningSystem, self).__init__()
        self.net = net
        self.lr = cfg.train.lr['G']
        self.experiment = experiment
        self.epoch = cfg.train.epoch
        self.r_factor = cfg.vae.r_factor
        self.step = 0

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epoch, eta_min=0)

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img = batch
        out, mu, log_var = self.net(img)
        loss = KL_R_Loss(out, img, mu, log_var, self.r_factor)
        self.experiment.log_metric('train/loss', loss.item(), step=batch_idx)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # Generative Images
        n_to_show = 16
        znew = np.random.normal(size = (n_to_show, self.net.z_dim))
        znew = torch.as_tensor(znew, dtype=torch.float32).cuda()

        gen_img = self.net.decoder(znew)
        gen_img = gen_img * 255

        joined_images_tensor = make_grid(gen_img, nrow=4, padding=2)

        joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
        joined_images = np.transpose(joined_images, [1,2,0])

        self.experiment.log_image(joined_images, name='genarative_img', step=self.step, image_channels='last')

        self.step += 1

        return None


# DCGAN - Lightning Module ---------------------------------------------------------------------------
class DCGAN_LightningSystem(pl.LightningModule):
    def __init__(self, G, D, cfg, experiment, checkpoint_path=None):
        super(DCGAN_LightningSystem, self).__init__()
        self.G = G
        self.D = D
        self.lr = cfg.train.lr
        self.experiment = experiment
        self.epoch = cfg.train.epoch
        self.z_dim = cfg.train.z_dim
        self.checkpoint_path = checkpoint_path
        self.cnt_train_step = 0
        self.step = 0
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def configure_optimizers(self):
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr['G'])
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr['D'])
        self.g_scheduler = CosineAnnealingLR(self.g_optimizer, T_max=self.epoch, eta_min=0)
        self.d_scheduler = CosineAnnealingLR(self.d_optimizer, T_max=self.epoch, eta_min=0)

        return [self.d_optimizer, self.g_optimizer], [self.d_scheduler, self.g_scheduler]

    def training_step(self, batch, batch_idx, optimizer_idx):
        img = batch
        b = img.size()[0]
        valid = torch.ones((b, 1)).cuda()
        fake = torch.zeros((b, 1)).cuda()

        # Train Discriminator
        if optimizer_idx == 0:
            z = torch.randn(b, self.z_dim).cuda()
            d_true_out = self.D(img)
            true_D_loss = self.criterion(d_true_out, valid)

            d_fake_out = self.D(self.G(z))
            fake_D_loss = self.criterion(d_fake_out, fake)

            D_loss = (true_D_loss + fake_D_loss) / 2

            self.experiment.log_metric('train/D_loss', D_loss.detach(), step=self.cnt_train_step)

            return {'loss': D_loss}

        # Train Generator
        elif optimizer_idx == 1:
            z = torch.randn(b, self.z_dim).cuda()
            g_fake_out = self.D(self.G(z))
            G_loss = self.criterion(g_fake_out, valid)
            self.experiment.log_metric('train/G_loss', G_loss.detach(), step=self.cnt_train_step)

            self.cnt_train_step += 1

            return {'loss': G_loss}

    def training_epoch_end(self, outputs):
        # Generative Images
        n_to_show = 16
        znew = np.random.normal(size = (n_to_show, self.z_dim))
        znew = torch.as_tensor(znew, dtype=torch.float32).cuda()

        gen_img = self.G(znew)
        # Reverse Normalization
        gen_img = gen_img * 0.5 + 0.5
        gen_img = gen_img * 255

        joined_images_tensor = make_grid(gen_img, nrow=4, padding=2)

        joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
        joined_images = np.transpose(joined_images, [1,2,0])

        self.experiment.log_image(joined_images, name='genarative_img', step=self.step, image_channels='last')

        self.step += 1

        # Save checkpoints
        if self.checkpoint_path is not None:
            checkpoint_paths = glob.glob(os.path.join(self.checkpoint_path, 'gan*'))
            for path in checkpoint_paths:
                self.experiment.log_asset(file_data=path, copy_to_tmp=True, overwrite=False)

        return None


# WGAN-GP - Lightning Module ---------------------------------------------------------------------------
class WGAN_GP_LightningSystem(pl.LightningModule):
    def __init__(self, G, D, cfg, experiment, checkpoint_path=None):
        super(WGAN_GP_LightningSystem, self).__init__()
        self.G = G
        self.D = D
        self.lr = cfg.train.lr
        self.cfg = cfg
        self.experiment = experiment
        self.checkpoint_path = checkpoint_path
        self.cnt_train_step = 0

    def configure_optimizers(self):
        # スケジューラーを入れるとモード崩壊起こしてしまうかも
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.lr['G'], betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.lr['D'], betas=(0.5, 0.999))

        return [self.d_optimizer, self.g_optimizer], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        img = batch
        b = img.size()[0]
        z = torch.randn(b, self.cfg.train.z_dim).cuda()

        # Train Discriminator
        if optimizer_idx == 0:
            d_true_out = self.D(img)
            true_D_loss = torch.mean(d_true_out)

            fake_img = self.G(z)
            d_fake_out = self.D(fake_img.detach())
            fake_D_loss = -1 * torch.mean(d_fake_out)

            # Interpolated
            alpha = torch.rand(b, 1, 1, 1).cuda()
            interpolated = (alpha * img + (1 - alpha) * fake_img.detach()).requires_grad_(True)
            interpolated_out = self.D(interpolated)

            gradient_criterion = GradientPaneltyLoss()
            gradient_loss = gradient_criterion(interpolated_out, interpolated)

            D_loss = true_D_loss + fake_D_loss + self.cfg.wgan_gp.gradientloss_weight * gradient_loss
            self.log('train/D_loss_valid', true_D_loss, on_epoch=True)
            self.log('train/D_loss_fake', fake_D_loss, on_epoch=True)
            self.log('train/D_loss_gradient', gradient_loss, on_epoch=True)
            self.log('train/D_loss', D_loss, on_epoch=True)

            return {'loss': D_loss}

        # Train Generator
        elif optimizer_idx == 1:
            g_fake_out = self.D(self.G(z))
            G_loss = torch.mean(g_fake_out)
            self.log('train/G_loss', G_loss, on_epoch=True)

            # Update Counter
            self.cnt_train_step += 1

            return {'loss': G_loss}

    def training_epoch_end(self, outputs):
        # Generative Images
        n_to_show = 16
        znew = np.random.normal(size = (n_to_show, self.cfg.train.z_dim))
        znew = torch.as_tensor(znew, dtype=torch.float32).cuda()

        gen_img = self.G(znew)
        # Reverse Normalization
        gen_img = gen_img * 0.5 + 0.5
        gen_img = gen_img * 255

        joined_images_tensor = make_grid(gen_img, nrow=4, padding=2)

        joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
        joined_images = np.transpose(joined_images, [1,2,0])

        self.logger.experiment.log_image(joined_images, name='genarative_img', step=self.current_epoch, image_channels='last')

        del joined_images, joined_images_tensor

        # Save checkpoints
        if self.checkpoint_path is not None:
            checkpoint_paths = sorted(glob.glob(os.path.join(self.checkpoint_path, 'wgan-gp*')))
            self.logger.experiment.log_asset(file_data=checkpoint_paths[-1], overwrite=True)

        return None


# CycleGAN - Lightning Module ---------------------------------------------------------------------------
class CycleGAN_LightningSystem(pl.LightningModule):
    def __init__(self, G_basestyle, G_stylebase, D_base, D_style, transform, experiment, cfg, checkpoint_path=None):
        super(CycleGAN_LightningSystem, self).__init__()
        self.G_basestyle = G_basestyle
        self.G_stylebase = G_stylebase
        self.D_base = D_base
        self.D_style = D_style
        self.lr = cfg.train.lr
        self.transform = transform
        self.reconstr_w = cfg.cyclegan.reconstr_w
        self.id_w = cfg.cyclegan.id_w
        self.experiment = experiment
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.cnt_train_step = 0
        self.cnt_epoch = 0

        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()

    def configure_optimizers(self):
        self.g_basestyle_optimizer = optim.Adam(self.G_basestyle.parameters(), lr=self.lr['G'], betas=(0.5, 0.999))
        self.g_stylebase_optimizer = optim.Adam(self.G_stylebase.parameters(), lr=self.lr['G'], betas=(0.5, 0.999))
        self.d_base_optimizer = optim.Adam(self.D_base.parameters(), lr=self.lr['D'], betas=(0.5, 0.999))
        self.d_style_optimizer = optim.Adam(self.D_style.parameters(), lr=self.lr['D'], betas=(0.5, 0.999))

        return [self.g_basestyle_optimizer, self.g_stylebase_optimizer, self.d_base_optimizer,
                self.d_style_optimizer], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        base_img, style_img = batch
        b = base_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).cuda()
        fake = torch.zeros(b, 1, 30, 30).cuda()

        # Count up
        self.cnt_train_step += 1

        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            # Validity
            # MSELoss
            val_base = self.generator_loss(self.D_base(self.G_stylebase(style_img)), valid)
            val_style = self.generator_loss(self.D_style(self.G_basestyle(base_img)), valid)
            val_loss = (val_base + val_style) / 2

            # Reconstruction
            reconstr_base = self.mae(self.G_stylebase(self.G_basestyle(base_img)), base_img)
            reconstr_style = self.mae(self.G_basestyle(self.G_stylebase(style_img)), style_img)
            reconstr_loss = (reconstr_base + reconstr_style) / 2

            # Identity
            id_base = self.mae(self.G_stylebase(base_img), base_img)
            id_style = self.mae(self.G_basestyle(style_img), style_img)
            id_loss = (id_base + id_style) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss

            logs = {'loss': G_loss, 'validity': val_loss, 'reconstr': reconstr_loss, 'identity': id_loss}
            # self.experiment.log_metrics(logs, step=self.cnt_train_step)

            return logs

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:
            # MSELoss
            D_base_gen_loss = self.discriminator_loss(self.D_base(self.G_stylebase(style_img)), fake)
            D_style_gen_loss = self.discriminator_loss(self.D_style(self.G_basestyle(base_img)), fake)
            D_base_valid_loss = self.discriminator_loss(self.D_base(base_img), valid)
            D_style_valid_loss = self.discriminator_loss(self.D_style(style_img), valid)

            D_gen_loss = (D_base_gen_loss + D_style_gen_loss) / 2

            # Loss Weight
            D_loss = (D_gen_loss + D_base_valid_loss + D_style_valid_loss) / 3

            logs = {'loss': D_loss}
            # self.experiment.log_metric('D_loss', D_loss, step=self.cnt_train_step)

            return logs

    def training_epoch_end(self, outputs):
        self.cnt_epoch += 1

        avg_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().detach() / 4 for i in range(4)])
        G_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().detach() / 2 for i in [0, 1]])
        D_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().detach() / 2 for i in [2, 3]])
        validity = sum([torch.stack([x['validity'] for x in outputs[i]]).mean().detach() / 2 for i in [0, 1]])
        reconstr = sum([torch.stack([x['reconstr'] for x in outputs[i]]).mean().detach() / 2 for i in [0, 1]])
        identity = sum([torch.stack([x['identity'] for x in outputs[i]]).mean().detach() / 2 for i in [0, 1]])

        logs = {
            'avg_loss': avg_loss, 'G_mean_loss': G_mean_loss, 'D_mean_loss': D_mean_loss,
            'validity': validity, 'reconstr': reconstr, 'identity': identity
        }

        self.experiment.log_metrics(logs, epoch=self.cnt_epoch)

        if self.cnt_epoch % 1 == 0:
            # Display Model Output
            target_img_paths = glob.glob('./data/CelebA-HQ-img/*.jpg')[:8]
            target_imgs = [self.transform(Image.open(path), phase='test') for path in target_img_paths]
            target_imgs = torch.stack(target_imgs, dim=0)
            target_imgs = target_imgs.cuda()

            gen_imgs = self.G_basestyle(target_imgs)
            gen_img = torch.cat([target_imgs, gen_imgs], dim=0)

            # Reverse Normalization
            gen_img = gen_img * 0.5 + 0.5
            gen_img = gen_img * 255

            joined_images_tensor = make_grid(gen_img, nrow=4, padding=2)

            joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
            joined_images = np.transpose(joined_images, [1, 2, 0])

            self.experiment.log_image(joined_images, name='output_img', step=self.cnt_epoch, image_channels='last')

            # Save checkpoints
            if self.checkpoint_path is not None:
                model = self.G_basestyle
                weight_name = f'weight_epoch_{self.cnt_epoch}.pth'
                weight_path = os.path.join(self.checkpoint_path, weight_name)
                torch.save(model.state_dict(), weight_path)
                time.sleep(3)
                self.experiment.log_asset(file_data=weight_path)
                os.remove(weight_path)
        else:
            pass

        return None