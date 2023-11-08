import torch
import numpy as np
import logging
from torch.utils.data import IterableDataset

from typing import List
from kaldiio import ReadHelper
from tqdm.auto import tqdm




def get_uniq_slice():
    info = torch.utils.data.get_worker_info()
    if info is None:
        local_id = 0
        local_nw = 1
    else:
        local_id = info.id
        local_nw = info.num_workers

    if torch.distributed.is_initialized():
        ddp_rank = torch.distributed.get_rank()
        ddp_nw = torch.distributed.get_world_size()
    else:
        ddp_rank = 0
        ddp_nw = 1
    start = local_id + (ddp_rank * local_nw)
    step = ddp_nw * local_nw
    return slice(start, None, step)




class FeatsIterableDataset(IterableDataset):
    def __init__(self, feats_rspecifiers: List[str], targets_rspecifier=None, shuffle=False, bos_id=None, eos_id=None):
        self.feats_rspecifiers = feats_rspecifiers
        self.targets_rspecifier = targets_rspecifier
        self.shuffle = shuffle
        self.bos_id = bos_id
        self.eos_id = eos_id
        if targets_rspecifier is not None:
            logging.info(f"Loading targets from {targets_rspecifier}")
            with ReadHelper(self.targets_rspecifier) as f:
                self.utt2target = {utt: torch.from_numpy(target) for utt, target in tqdm(f, desc='Loading targets...')}
        else:
            self.utt2target = None

    def __len__(self):
        return len(self.utt2target)

    def items(self):
        #rspecs = self.feats_rspecifiers[info.id: len(self.feats_rspecifiers): info.num_workers]
        rspecs = self.feats_rspecifiers[get_uniq_slice()]
        if self.shuffle:
            rspecs = np.random.permutation(rspecs)
        for rspec in rspecs:
            logging.info(f'Processing {rspec}')
            with ReadHelper(rspec) as f:
                for uid, feats in f:
                    if self.utt2target is not None:
                        targets = self.utt2target.get(uid)
                        tgt_key_padding_mask = torch.zeros(targets.shape[0])
                        targets_len = targets.shape[0]
                    else:
                        targets = None
                        tgt_key_padding_mask = None
                        targets_len = None

                    yield {'uid': uid,
                           'feats': torch.as_tensor(feats, dtype=torch.float32),
                           'feats_len': feats.shape[0],
                           'targets': targets,
                           'targets_len': targets_len,
                           'src_key_padding_mask': torch.zeros(feats.shape[0]),
                           'tgt_key_padding_mask': tgt_key_padding_mask,
                          }

    def __iter__(self):
        return iter(self.items())

    def collate(self, batch):
        collated_batch = dict()
        collated_batch['uids'] = [e['uid'] for e in batch]
        collated_batch['feats'] = torch.nn.utils.rnn.pad_sequence([e['feats'] for e in batch],
                                                batch_first=False,
                                                padding_value=1.0)
        collated_batch['feats_len'] = torch.as_tensor([e['feats_len'] for e in batch], dtype=torch.long)
        collated_batch['src_key_padding_mask'] = torch.nn.utils.rnn.pad_sequence([e['src_key_padding_mask'] for e in batch],
                                                               batch_first=True,
                                                               padding_value=True)
        if batch[0]['targets'] is not None:
            collated_batch['targets'] = torch.concatenate([e['targets'] for e in batch]) # (SumTime,)
            collated_batch['tgt_key_padding_mask'] = torch.nn.utils.rnn.pad_sequence(
                                                [e['tgt_key_padding_mask'] for e in batch],
                                                batch_first=True,
                                                padding_value=True)
            collated_batch['targets_len'] = torch.as_tensor([e['targets_len'] for e in batch], dtype=torch.long)
        else:
            collated_batch['targets'] = None
            collated_batch['tgt_key_padding_mask'] = None
            collated_batch['targets_len'] = None
        return collated_batch

