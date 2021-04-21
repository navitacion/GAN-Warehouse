# GAN Warehouse

## Get Started

### Load Data

```commandline
# CelebA-HQ-Dataset
bash download.sh celeba-hq-dataset

# Animal Faces HQ
bash download.sh afhq-dataset
```

### Env

if you use "Comet ML", you should make ".env" file based on ".env.sample"


### Run Train

Build Docker Container

```commandline
docker-compose up --build -d
docker exec -it gan_env bash
```

Then, Train Model

```commandline
python train.py
```