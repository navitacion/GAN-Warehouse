import os
import glob
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from src.lightning import VAE_LightningSystem, DCGAN_LightningSystem, WGAN_GP_LightningSystem, CycleGAN_LightningSystem, SAGAN_LightningSystem, PROGAN_LightningSystem
from src.lightning import CycleGANDataModule, SingleImageDataModule
from src.utils.augment import ImageTransform

from src.models.build import build_model


@hydra.main('config.yml')
def main(cfg: DictConfig):
    print(f'Training {cfg.train.model} Model')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    # Data Augmentation  --------------------------------------------------------
    transform = ImageTransform(cfg) if cfg.train.model != 'progan' else None

    # DataModule  ---------------------------------------------------------------
    dm = None
    data_dir = './data'
    if cfg.train.data == 'celeba_hq':
        img_paths = glob.glob(os.path.join(data_dir, 'celeba_hq', '**/*.jpg'), recursive=True)
        dm = SingleImageDataModule(img_paths, transform, cfg)

    elif cfg.train.data == 'afhq':
        img_paths = glob.glob(os.path.join(data_dir, 'afhq', '**/*.jpg'), recursive=True)
        dm = SingleImageDataModule(img_paths, transform, cfg)

    elif cfg.train.data == 'ffhq':
        img_paths = glob.glob(os.path.join(data_dir, 'ffhq', '**/*.png'), recursive=True)
        dm = SingleImageDataModule(img_paths, transform, cfg)

    # Model  --------------------------------------------------------------------
    nets = build_model(cfg.train.model, cfg)

    # Comet_ml  -----------------------------------------------------------------
    load_dotenv('.env')
    logger = CometLogger(api_key=os.environ['COMET_ML_API_KEY'],
                         project_name=os.environ['COMET_ML_PROJECT_NAME'],
                         experiment_name=f"{cfg.train.model}")

    logger.log_hyperparams(dict(cfg.train))


    # Lightning Module  ---------------------------------------------------------
    model = None
    checkpoint_path = 'checkpoints/'
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, filename='{epoch:02d}', prefix=cfg.train.model, period=1)

    if cfg.train.model == 'vae':
        model = VAE_LightningSystem(nets[0], cfg)

    elif cfg.train.model == 'dcgan':
        model = DCGAN_LightningSystem(nets[0], nets[1], cfg, checkpoint_path)

    elif cfg.train.model == 'wgan_gp':
        logger.log_hyperparams(dict(cfg.wgan_gp))
        model = WGAN_GP_LightningSystem(nets[0], nets[1], cfg, checkpoint_path)

    elif cfg.train.model == 'cyclegan':
        logger.log_hyperparams(dict(cfg.cyclegan))
        data_dir = 'data/'
        base_img_paths = glob.glob(os.path.join(data_dir, cfg.cyclegan.base_imgs_dir, '**/*.jpg'), recursive=True)
        style_img_paths = glob.glob(os.path.join(data_dir, cfg.cyclegan.style_imgs_dir, '**/*.jpg'), recursive=True)
        dm = CycleGANDataModule(base_img_paths, style_img_paths, transform, cfg, phase='train', seed=cfg.train.seed)
        model = CycleGAN_LightningSystem(nets[0], nets[1], nets[2], nets[3],
                                         transform, cfg, checkpoint_path)

    elif cfg.train.model == 'sagan':
        logger.log_hyperparams(dict(cfg.sagan))
        model = SAGAN_LightningSystem(nets[0], nets[1], cfg, checkpoint_path)

    elif cfg.train.model == 'progan':
        logger.log_hyperparams(dict(cfg.progan))
        model = PROGAN_LightningSystem(nets[0], nets[1], cfg, checkpoint_path)

    # Trainer  ---------------------------------------------------------
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        gpus=1,
        callbacks=[checkpoint_callback],
        # fast_dev_run=True,
        # resume_from_checkpoint='./checkpoints/sagan-epoch=11.ckpt'
    )

    # Train
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()