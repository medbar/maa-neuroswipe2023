import torch
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from torch import nn
from glob import glob
from tqdm.auto import tqdm
from torchaudio import transforms as T
import pytorch_lightning as pl
from maatool.data.feats_itdataset_v2 import FeatsIterableDatasetV2
from maatool.models.transformer import TransformerWithSinPos


def get_new_tgt(prev_tgt, hyp_logprobs, logits, topk=4):
    """
    prev_tgt - (T, N_hyp)
    hyp_logprobs - (N_hyp, )
    logits - (N_hyp, C)
    """
    assert len(prev_tgt.shape) == 2, f"{prev_tgt.shape=}"
    assert len(hyp_logprobs.shape) == 1, f"{hyp_logprobs.shape=}"
    assert len(logits.shape) == 2, f"{logits.shape=}"
    assert prev_tgt.shape[1] == hyp_logprobs.shape[0] == logits.shape[0], (
        f"{prev_tgt.shape=} {hyp_logprobs.shape=} {logits.shape=}"
    )

    nt_topk_logits, nt_topk_idx = logits.topk(k=topk, axis=-1)
    #print("nt_topk_idx", nt_topk_idx, nt_topk_idx.shape)
    # (N, K)
    next_tokens = nt_topk_idx.T.reshape(1, -1)
    #print("next_tokens", next_tokens, next_tokens.shape)
    # (1, N*(repeat k times))
    # (T, N*(repeat k times))
    new_hyp_tgt = torch.concatenate([prev_tgt.repeat(1, topk), next_tokens], axis=0)
    #print(f"{new_hyp_tgt=}", new_hyp_tgt.shape)
    # (T+1, N*(repeat k times))
    new_scores = nt_topk_logits.T.reshape(-1)
    # N*(repeat k times)
    prew_scores = hyp_logprobs.repeat(topk)
    #print("prew_scores", prew_scores)
    # N*(repeat k times)
    new_hyp_logprob = prew_scores + new_scores
    #print("new_hyp_logprob", new_hyp_logprob)
    new_hyp_logprob, idx = new_hyp_logprob.topk(k=topk)
    #print(idx)
    new_hyps = new_hyp_tgt[:, idx]
    #print("new_hyps", new_hyps, new_hyp_logprob)
    # (T+1, N*k), (N,)
    return new_hyps, new_hyp_logprob


def test_get_new_tgt():
    tgt, logits = get_new_tgt(torch.LongTensor([[1,]]), torch.tensor([-1.]), torch.tensor([[-3, -4, -7, -2, -5]]))
    print(">>>>\n", tgt, logits)
    get_new_tgt(tgt, logits, torch.tensor([[100,    110,  200],
                                           [100,    110,  200],
                                           [100,    110,  200],
                                           [100,    110,  200]]), topk=2)

def sep_ready_tgt(tgt, logprobs, eos_id=2):
    """
    tgt - (T, N)
    logprobs - (N,)
    """
    assert tgt.shape[1] == logprobs.shape[0], (
        f"{tgt.shape=} {logprobs.shape=}"
    )

    is_end_mask = ((tgt == eos_id).sum(axis=0) > 0)
    # (N,)
    #print(is_end_mask)
    ready_tgt = tgt[:, is_end_mask]
    ready_logprobs = logprobs[is_end_mask]

    ready_list = [(l.cpu().item(), t.cpu().tolist()) for l, t in zip(ready_logprobs, ready_tgt.T)]

    not_ready_tgt = tgt[:, ~is_end_mask]
    not_ready_logprobs = logprobs[~is_end_mask]
    assert not_ready_tgt.shape[1] == not_ready_logprobs.shape[0], (
        f"{not_ready_tgt.shape[1]=} {not_ready_logprobs.shape[0]=}"
    )
    return ready_list, not_ready_tgt, not_ready_logprobs


def test_set_ready_tgt():
    print(sep_ready_tgt(torch.LongTensor([[1, 1], [2, 3]]), torch.tensor([-1., -3])))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

class SwipeTransformerRecognizer(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-4, speed=42):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.backbone = backbone
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        set_random_seed(speed)

    def forward(self, feats, **kwargs):
        # (T, N, E)
        return self.backbone(feats, **kwargs)

    def get_loss(self, batch):
        # batch - (Time, Batch, ...)
        feats = batch['feats']
        # (Time, Batch, num_feats)
        tgt = batch['targets'][:-1]
        tgt_key_padding_mask = batch['tgt_key_padding_mask'][:, 1:]
        # (Batch, Seq-1)

        logits = self.forward(feats=feats,
                              tgt=tgt,
                              src_key_padding_mask=batch['src_key_padding_mask'],
                              tgt_key_padding_mask=tgt_key_padding_mask)
        # (Seq-1, Batch, C)
        S, N, C = logits.shape
        targets = batch['targets'][1:]
        # (Seq-1, Batch)
        # print("loss ", logits.shape, targets.shape)
        loss = self.ce_loss(logits.view(-1, C), targets.reshape(-1))

        return loss


    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True,  batch_size=len(batch['uids']))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, on_step=True,  batch_size=len(batch['uids']))

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('test_loss', loss,  batch_size=len(batch['uids']))

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

    def predict_topk(self, dl, tokenizer, topk=4, bos_id=1, eos_id=2, max_out_len=26, device='cuda'):
        self.eval()
        utt2word= defaultdict(list)
        utt2logs = defaultdict(list)
        pbar = tqdm(dl)
        with torch.no_grad():
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                memory = self.backbone.forward_encoder(batch['feats'],
                                                  src_key_padding_mask=batch['src_key_padding_mask'])
                assert memory.shape[1] == 1, f"{memory.shape=}"
                # (SrcTime, Batch, E)
                tgt = torch.full(size=(1, 1),
                                 fill_value=bos_id,
                                 dtype=torch.long,
                                 device=memory.device)
                hyp_logprobs = torch.zeros((1), device=memory.device)
                tgt_ready = []
                mkpm = batch['src_key_padding_mask']
                for l in range(max_out_len):
                    #print(f"{tgt.shape=}")
                    tgt_logits = self.backbone.forward_decoder(tgt,
                                                        memory.repeat(1, tgt.shape[1], 1),
                                                        memory_key_padding_mask=mkpm.repeat((tgt.shape[1], 1)))
                    tgt_logits = tgt_logits.log_softmax(dim=-1)

                    new_tgt, logprobs = get_new_tgt(tgt, hyp_logprobs, tgt_logits[-1], topk=topk)
                    ready, tgt, hyp_logprobs = sep_ready_tgt(new_tgt, logprobs)
                    tgt_ready.extend(ready)
                    if len(tgt_ready) >= topk:
                        break

                uid = batch['uids'][0]
                if len(tgt_ready) == 0:
                    logging.warning(f"tgt_ready is 0 for {uid}. {tgt.shape=}. Use all hyps as ready hyps")
                    tgt_ready = [(l.cpu().item(), t.cpu().tolist()) for l, t in zip(hyp_logprobs, tgt.T)]

                out_indices = []
                for logprob, indices in sorted(tgt_ready, reverse=True):
                    joined = tokenizer.decode(indices) #.split()[0]
                    utt2word[uid].append(joined)
                    utt2logs[uid].append(logprob)
                d = '|'+'|'.join(utt2word[uid]) + "|"
                pbar.set_description(f"{d}".ljust(40, '=')[:40], refresh=False)
        return utt2word, utt2logs
