import torch
import numpy as np
import random
import pandas as pd
from torch import nn
from glob import glob
from tqdm.auto import tqdm
from torchaudio import transforms as T


from maatool.data.feats_itdataset_v2 import FeatsIterableDatasetV2
#from maatool.models.transformer import TransformerWithSinPos
from maatool.models.conformer import ConformerWithSinPos
from maatool.lightning.swipe_recognizer import SwipeTransformerRecognizer, set_random_seed

import pytorch_lightning as pl

from collections import defaultdict



import logging
import logging.config


print("Start training. Cuda ", torch.cuda.is_available())

def configure_logging(log_level):
    handlers =  {
        "maa": {
            "class": "logging.StreamHandler",
            "formatter": "maa_basic",
            "stream": "ext://sys.stdout",
        }
    }
    CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"maa_basic": {"format": '%(asctime)s %(name)s %(pathname)s:%(lineno)d - %(levelname)s - %(message)s'}},
        "handlers": handlers,
        "loggers": {"maa": {"handlers": handlers.keys(), "level": log_level}},
        "root": {"handlers": handlers.keys(), "level": log_level}
    }
    logging.config.dictConfig(CONFIG)
configure_logging("DEBUG")


model = ConformerWithSinPos(feats_dim=37, num_tokens=500, num_decoder_layers=8, num_encoder_layers=8)
#pl_module = SwipeTransformerRecognizer(model)
v_13_ckpt = 'exp/models/conformer_v1/lightning_logs/version_50454211/checkpoints/last.ckpt'
pl_module = SwipeTransformerRecognizer.load_from_checkpoint(v_13_ckpt,
                                                            backbone=model,
                                                            map_location='cpu')


trainer = pl.Trainer(max_epochs=1, log_every_n_steps=400, reload_dataloaders_every_n_epochs=1,
                     default_root_dir='exp/models/conformer_v1.11',
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=100),
                                pl.callbacks.ModelCheckpoint(every_n_train_steps=1000,
                                                             save_last=True)],
                     accumulate_grad_batches=6,
                     check_val_every_n_epoch=10,
                     val_check_interval=20000)

set_random_seed(43)

val_ds = FeatsIterableDatasetV2([f"ark:data_feats/valid/feats.ark"],
                                                             targets_rspecifier='ark:exp/bpe500/valid-text.int',
                                                                shuffle=False,
                                                               bos_id=1,
                                                               eos_id=2,
                                                               batch_first=False)
val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=1, collate_fn=val_ds.collate_pad)

train_ds = FeatsIterableDatasetV2([f"ark:{f}" for f in sorted(["data_feats/train/feats.2.ark",
                                                                   "data_feats/train/feats.3.ark",
                                                                   "data_feats/train/feats.4.ark",
                                                                   "data_feats/train/feats.5.ark",
                                                                   "data_feats/train/feats.6.ark",
                                                                   "data_feats/train/feats.7.ark",
                                                                   "data_feats/train/feats.8.ark",
                                                                   "data_feats/train/feats.9.ark"])],
                                                                    targets_rspecifier='ark:exp/bpe500/train-text.int.ark',
                                                                    shuffle=True,
                                                                    bos_id=1,
                                                                    eos_id=2,
                                                                   batch_first=False)
#train_ds=val_ds

train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=16, collate_fn=train_ds.collate_pad,
                                               num_workers=0)



#result = trainer.test(pl_module, val_dataloader)
#print(result)

print("Train")
trainer.fit(pl_module, train_dataloader, val_dataloader)
#trainer.fit(pl_module, train_dataloader)

result = trainer.test(pl_module, val_dataloader)
print(result)

