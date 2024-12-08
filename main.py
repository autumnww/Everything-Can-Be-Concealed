from options.train_options import TrainOptions
from models.chameleonNet_model import DataModule, ChameleonNetModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import torch
import numpy as np
def main():
    opt = TrainOptions().parse()   # get training 
    dataloader = DataModule(opt)
    model = ChameleonNetModel(opt)

    if opt.is_train:
        torch.manual_seed(6)
        torch.cuda.manual_seed_all(6)
        np.random.seed(6)

        checkpoint_callback = ModelCheckpoint(
            dirpath=opt.name,
            filename='model-{epoch:02d}',
            save_top_k=-1, 
            every_n_epochs=5, 
            save_last=True,
            save_weights_only=True,
        )

        trainer = Trainer(
            accelerator='gpu', 
            devices=[0],
            max_epochs=30,
            strategy='ddp_find_unused_parameters_true',
            callbacks=checkpoint_callback,
        )
        trainer.fit(model, dataloader)
    else:
        trainer = Trainer(
            accelerator='gpu', 
            devices=[0,1]
        )
        model = ChameleonNetModel.load_from_checkpoint(opt.checkpoints_dir + opt.name, opt=opt)
        trainer.test(model, dataloader)
        
if __name__ == "__main__":
    main()