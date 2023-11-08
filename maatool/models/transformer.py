import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])



class TransformerEncoder(nn.Module):
    def __init__(self, feats_dim, num_tokens, d_model=512, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, nhead=8, max_len=400):
        super().__init__()
        self.feats_dim = feats_dim
        self.num_tokens = num_tokens
        self.max_len = max_len
        self.d_model = d_model
        self.input_ff = nn.Linear(feats_dim, d_model)
        self.tgt_embedding = nn.Embedding(num_tokens, d_model)
        self.positional_encoding = PositionalEncoding(dim_model=d_model, dropout_p=dropout, max_len=max_len)
        #layer = torch.nn.TransformerEncoderLayer(d_model=dim,
        #                                         nhead=nhead,
        #                                         dim_feedforward=ff_dim,
        #                                         dropout=dropout,
        #                                         batch_first=False)
        self.transformer = torch.nn.Transformer(d_model=dim,
                                                nhead=nhead,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=False)
        self.head = nn.Linear(d_model, num_tokens)
        #torch.nn.Transformer(encoder_layer=layer, num_layers=num_layers)

    def forward(self, feats, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_is_causal=True, **kwargs):
        """
        feats - (Batch, Time, num_feats)
        tgt - (Batch, Time)
        src_key_padding_mask - (Batch, Time)
        tgt_key_padding_mask - (Batch, Time)
        """
        pass

        src_embs = self.input_ff(feats)
        src_embs = self.positional_encoding(src_embs)
        tgt_embs = self.tgt_embedding(tgt) * math.sqrt(self.dim_model)
        T, B, E = embs.shape
        # print(embs.shape)
        src_embs = src_embs.permute(1, 0, 2)
        tgt_embs = tgt_embs.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        embs = self.transformer(src_embs, tgt_embs,
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                tgt_is_causal=tgt_is_causal)
        out = self.head(embs)
        return embs  # (Time, Batch, Classes)
