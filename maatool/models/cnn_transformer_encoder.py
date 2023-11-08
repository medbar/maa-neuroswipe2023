import torch
from torch import nn


class CNNTransformerEncoderWithPosEncoding(nn.Module):
    def __init__(self, feats_dim, out_dim, dim=512, num_layers=6, ff_dim=2048, dropout=0.1, nhead=4, max_len=400):
        super().__init__()
        self.feats_dim = feats_dim
        self.max_len = max_len
        #self.input_ff = nn.Linear(feats_dim, dim)
        self.input_cnn = nn.Sequential(*[torch.nn.Conv1d(feats_dim, dim//4, 3, 1, 1),
                     torch.nn.BatchNorm1d(dim//4),
                     torch.nn.ReLU(),
                     torch.nn.Conv1d(dim//4, dim//4, 3, 2, 1),
                     torch.nn.BatchNorm1d(dim//4),
                     torch.nn.ReLU(),
                     torch.nn.Conv1d(dim//4, dim//2, 3, 1, 1),
                     torch.nn.BatchNorm1d(dim//2),
                     torch.nn.ReLU(),
                     torch.nn.Conv1d(dim//2, dim, 5, 2, 2),
                     torch.nn.BatchNorm1d(dim),
                     torch.nn.ReLU()])

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
        # ( T, N, C)
        feats = feats.permute(1, 2, 0)
        # (N, C, T)
        embs = self.input_cnn(feats)
        embs = embs.permute(2, 0, 1)
        T, B, E = embs.shape
        # print(embs.shape)
        pos = torch.arange(0, min(T, self.max_len - 1), device=embs.device).repeat(B, 1).T
        pos_embs = self.positional_encoding(pos)
        embs = embs + pos_embs
        #embs = self.encoder(embs, src_key_padding_mask=src_key_padding_mask)
        embs = self.encoder(embs)

        logits = self.head(embs)
        return logits  # (Time, Batch, Phones)
