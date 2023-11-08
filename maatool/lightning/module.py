import pytorch_lightning as pl
import torch


import torch.nn.functional

class NeuroSwipePL(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.ctc_criterion =

