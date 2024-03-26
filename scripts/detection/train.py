import torch
import wandb
import transformers as tr
import lightning.pytorch as pl
from pytorch_lightning.utilities import rank_zero_only

import gc
import yaml
from pathlib import Path
from argparse import ArgumentParser

from model import DetectionModel
from data import CityscapesDetectionDataModule


def read_args():
    parser = ArgumentParser()
    parser.add_argument('--path_config', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()


def recursive_eval(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            recursive_eval(v)
        else:
            try:
                d[k] = eval(v)
            except (SyntaxError, NameError, TypeError):
                pass
    return d


class UnfreezeCallback(pl.Callback):

    def __init__(self, unfreeze_epoch, lr, lr_backbone, weight_decay):
        self.unfreeze_epoch = unfreeze_epoch
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            # unfreeze the entire detection module
            pl_module.unfreeze()
            # update the optimizer
            param_dicts = [
                {"params": [p for n, p in trainer.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in trainer.model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.lr_backbone,
                },
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
            trainer.optimizers = [optimizer]

            print(f"[INFO] Unfreezing the entire model at epoch {self.unfreeze_epoch}.")
            

def main():
    args = read_args()
    DATASET_ROOT = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir) 
    
    print('[INFO] Loading configuration...')
    with open(args.path_config, 'r') as f:
        CONFIG = recursive_eval(yaml.safe_load(f))
    print('[INFO] Configuration loaded:')
    print(CONFIG)
    
    TRAIN_DICT = CONFIG['training']
    MODEL_DICT = TRAIN_DICT['model']

    # create datamodule and set it for training
    print('[INFO] Creating datamodule...')
    datamodule = CityscapesDetectionDataModule(
        root=DATASET_ROOT,
        annFiles={
            'train': DATASET_ROOT / 'annotations' / 'cityscapes_instances_train.json',
            'val': DATASET_ROOT / 'annotations' / 'cityscapes_instances_val.json',
            'test': DATASET_ROOT / 'annotations' / 'cityscapes_instances_test.json'
        },
        batch_size=TRAIN_DICT['batch_size'],
        num_workers=TRAIN_DICT['num_workers'],
        processor=tr.DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    )
    print('[INFO] Datamodule created.\n')
    
    # use logger from Weights & Biases
    wandb_logger = pl.loggers.WandbLogger(project=CONFIG['project'], log_model=True,
                                          save_dir=OUTPUT_DIR / 'artifacts' if OUTPUT_DIR else None)
    if rank_zero_only.rank == 0:
        print('[INFO] Setup Weights & Biases...')
        wandb_logger.experiment.config.update(TRAIN_DICT)
        print('[INFO] Weights & Biases setup complete.\n')

    # initialize model
    print('[INFO] Initializing model...')
    if TRAIN_DICT['resume_artifact']:
        artifact_dir = wandb_logger.download_artifact(
            artifact=TRAIN_DICT['resume_artifact'],
            save_dir=OUTPUT_DIR / 'artifacts' if OUTPUT_DIR else None,
            artifact_type='model')
        model = DetectionModel.load_from_checkpoint(Path(artifact_dir) / 'model.ckpt')
    else:
        model = DetectionModel(max_class_id=datamodule.MAX_CLASS_ID, lr=MODEL_DICT['lr'])
    print('[INFO] Model initialized.\n')
    
    unfreeze_callback = UnfreezeCallback(
        unfreeze_epoch=TRAIN_DICT['unfreeze_epoch'],
        lr=MODEL_DICT['lr'],
        lr_backbone=MODEL_DICT['lr_backbone'],
        weight_decay=MODEL_DICT['weight_decay'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss', 
        mode='min', 
        dirpath=OUTPUT_DIR / 'checkpoints' if OUTPUT_DIR else None)
    
    # initialize the trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=TRAIN_DICT['devices'],
        max_epochs=TRAIN_DICT['max_epochs'],
        logger=wandb_logger,
        callbacks=[unfreeze_callback, checkpoint_callback],
        log_every_n_steps=50,
        gradient_clip_val=0.1,
        overfit_batches=TRAIN_DICT['overfit_batches'])
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # keep track of the model including gradients
    wandb_logger.watch(model, log='all')
    
    # start training
    print('[INFO] Training...')
    trainer.fit(model, datamodule)
    print('[INFO] Training complete.\n')


if __name__ == '__main__':
    main()
