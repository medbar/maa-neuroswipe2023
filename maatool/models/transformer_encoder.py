import torch
from torch import nn


class TransformerEncoderWithPosEncoding(nn.Module):
    def __init__(self, feats_dim, out_dim, dim=512, num_layers=6, ff_dim=2048, dropout=0.1, nhead=4, max_len=400):
        super().__init__()
        self.feats_dim = feats_dim
        self.max_len = max_len
        self.input_ff = nn.Linear(feats_dim, dim)
        self.positional_encoding = nn.Embedding(max_len, dim)
        layer = torch.nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=nhead,
                                                   dim_feedforward=ff_dim,
                                                   dropout=dropout,
                                                   batch_first=False)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)

        self.head = nn.Linear(dim, out_dim)

    def forward(self, feats, src_key_padding_mask=None, **kwargs):
        pass
        #print(feats.shape)
        # (T, B, C)
        embs = self.input_ff(feats)
        T, B, E = embs.shape
        # print(embs.shape)
        pos = torch.arange(0, min(T, self.max_len - 1), device=feats.device).repeat(B, 1).T
        pos_embs = self.positional_encoding(pos)
        embs += pos_embs
        embs = self.encoder(embs, src_key_padding_mask=src_key_padding_mask)

        logits = self.head(embs)
        return logits  # (Time, Batch, Phones)
