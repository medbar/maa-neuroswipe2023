import torch
import logging
from torch.utils.data import IterableDataset

from typing import List
from kaldiio import ReadHelper
from tqdm.auto import tqdm


class FeatsIterableDataset:
    def __init__(self, feats_rspecifiers: List[str], targets_rspecifier=None):
        self.feats_rspecifiers = feats_rspecifiers
        self.targets_rspecifier = targets_rspecifier
        if targets_rspecifier is not None:
            logging.info(f"Loading targets from {targets_rspecifier}")
            with ReadHelper(self.targets_rspecifier) as f:
                self.utt2target = {utt: target for utt, target in tqdm(f)}
        else:
            self.utt2target = None

    def items(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            rspecs = self.feats_rspecifiers
        else:
            rspecs = self.feats_rspecifiers[info.id: len(self.feats_rspecifiers): info.num_workers]
        for rspec in rspecs:
            logging.debug(f'Processing {rspec}')
            with ReadHelper(rspec) as f:
                for uid, feats in f:
                    if self.utt2target is not None:
                        targets = self.utt2target.get(uid)
                        tgt_key_padding_mask = torch.zeros(targets.shape[0])
                    else:
                        targets = None
                        tgt_key_padding_mask = None

                    yield {'uid': uid,
                           'feats': feats,
                           'targets': targets,
                           'src_key_padding_mask': torch.zeros(feats.shape[0]),
                           'tgt_key_padding_mask': tgt_key_padding_mask}

    def __iter__(self):
        return iter(self.items())

    def collate(self, batch):
        collated_batch = dict()
        collated_batch['feats'] = torch.nn.utils.rnn.pad_sequence([e['feats'] for e in batch],
                                                batch_first=False,
                                                padding_value=1.0)
        collated_batch['src_key_padding_mask'] = torch.nn.utils.rnn.pad_sequence([e['src_key_padding_mask'] for e in batch],
                                                               batch_first=True,
                                                               padding_value=True)
        if self.utt2target is not None:
            collated_batch['targets'] = torch.concatenate([e['targets'] for e in batch])
            collated_batch['tgt_key_padding_mask'] = torch.nn.utils.rnn.pad_sequence(
                                                [e['tgt_key_padding_mask'] for e in batch],
                                                batch_first=True,
                                                padding_value=True)
        else:
            collated_batch['targets'] = None
            collated_batch['tgt_key_padding_mask'] = None
        return collated_batch

