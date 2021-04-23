import os
import glob
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from src.lightning import VAE_LightningSystem, DCGAN_LightningSystem, WGAN_GP_LightningSystem, CycleGAN_LightningSystem, SAGAN_LightningSystem
from src.lightning import CelebAHQDataModule, CycleGANDataModule
from src.utils.augment import ImageTransform

from src.models.build import build_model


@hydra.main('config.yml')
def main(cfg: DictConfig):
    print(f'Training {cfg.train.model} Model')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    # Data Augmentation  --------------------------------------------------------
    transform = ImageTransform(cfg)

    # Model  --------------------------------------------------------------------
    nets = build_model(cfg.train.model, cfg)

    # Comet_ml  -----------------------------------------------------------------
    load_dotenv('.env')
    logger = CometLogger(api_key=os.environ['COMET_ML_API_KEY'],
                         project_name=os.environ['COMET_ML_PROJECT_NAME'],
                         experiment_name=f"{cfg.train.model}")

    logger.log_hyperparams(dict(cfg.train))


    # Lightning Module  ---------------------------------------------------------
    dm = None
    model = None
    checkpoint_path = 'checkpoints/'
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, filename='{epoch:02d}', prefix=cfg.train.model, period=1)

    if cfg.train.model == 'vae':
        data_dir = 'data/'
        dm = CelebAHQDataModule(data_dir, transform, cfg)
        model = VAE_LightningSystem(nets[0], cfg)

    elif cfg.train.model == 'dcgan':
        data_dir = 'data/'
        dm = CelebAHQDataModule(data_dir, transform, cfg)
        cfg.train.img_size = 128
        model = DCGAN_LightningSystem(nets[0], nets[1], cfg, checkpoint_path)

    elif cfg.train.model == 'wgan_gp':
        logger.log_hyperparams(dict(cfg.wgan_gp))
        data_dir = 'data/'
        dm = CelebAHQDataModule(data_dir, transform, cfg)
        model = WGAN_GP_LightningSystem(nets[0], nets[1], cfg, checkpoint_path)

    elif cfg.train.model == 'cyclegan':
        logger.log_hyperparams(dict(cfg.cyclegan))
        data_dir = 'data/'
        base_img_paths = glob.glob(os.path.join(data_dir, 'celeba_hq', '**/*.jpg'), recursive=True)
        style_img_paths = glob.glob(os.path.join(data_dir, 'van_gogh_paintings', '**/*.jpg'), recursive=True)
        dm = CycleGANDataModule(base_img_paths, style_img_paths, transform, cfg, phase='train', seed=cfg.train.seed)
        model = CycleGAN_LightningSystem(nets[0], nets[1], nets[2], nets[3],
                                         transform, cfg, checkpoint_path)

    elif cfg.train.model == 'sagan':
        logger.log_hyperparams(dict(cfg.sagan))
        data_dir = 'data/'
        dm = CelebAHQDataModule(data_dir, transform, cfg)
        model = SAGAN_LightningSystem(nets[0], nets[1], cfg, checkpoint_path)

    # Trainer  ---------------------------------------------------------
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        gpus=1,
        callbacks=[checkpoint_callback],
        # resume_from_checkpoint='./checkpoints/epoch=30.ckpt'
    )

    # Train
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()