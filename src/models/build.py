from .vae import VAE
from .dcgan import Generator, Discriminator
from .wgan_gp import WGAN_GP_Generator, WGAN_GP_Discriminator
from .cycle_gan import CycleGAN_Unet_Generator_512, CycleGAN_Discriminator_512

from src.utils.utils import init_weights


def build_model(type, cfg):

    z_dim = cfg.train.z_dim
    img_size = cfg.train.img_size

    model_dict = {
        'vae': [VAE(z_dim)],
        'dcgan': [Generator(z_dim), Discriminator()],
        'wgan_gp': [WGAN_GP_Generator(z_dim, img_size), WGAN_GP_Discriminator(img_size)],
        'cyclegan':[
            CycleGAN_Unet_Generator_512(), CycleGAN_Unet_Generator_512(),
            CycleGAN_Discriminator_512(), CycleGAN_Discriminator_512()
        ]
    }

    models = model_dict[type]

    # init weight
    if type == 'cyclegan':
        models = [init_weights(net, init_type='normal') for net in models]


    return models
