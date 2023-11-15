import torch
import math
from torch import nn
from torchaudio.models import Conformer


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
        # pos_encoding - (Time, Embedding)
        pos_encoding = pos_encoding.unsqueeze(1) #.transpose(0, 1)
        # pos_encoding - (Time, 1, Embedding)
        print(f"PositionalEncoding shape is {pos_encoding.shape}")
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        # (Time, Batch, E)
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0)])



class ConformerWithSinPos(nn.Module):
    def __init__(self, feats_dim, num_tokens, d_model=512, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, nhead=8, max_len=400, max_out_len=30):
        super().__init__()
        self.feats_dim = feats_dim
        self.num_tokens = num_tokens
        self.max_len = max_len
        self.d_model = d_model
        self.input_ff = nn.Linear(feats_dim, d_model)
        self.tgt_embedding = nn.Embedding(num_tokens, d_model)
        self.positional_encoding = PositionalEncoding(dim_model=d_model, dropout_p=dropout, max_len=max_len)
        #self.decoder_pos_encoding = nn.Embedding(max_out_len, d_model)
        #layer = torch.nn.TransformerEncoderLayer(d_model=dim,
        #                                         nhead=nhead,
        #                                         dim_feedforward=ff_dim,
        #                                         dropout=dropout,
        #                                         batch_first=False)
        self.encoder = Conformer(input_dim=d_model,
                                 num_heads=nhead,
                                 ffn_dim=dim_feedforward,
                                 num_layers=num_encoder_layers,
                                 depthwise_conv_kernel_size=31,
                                 dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.head = nn.Linear(d_model, num_tokens)
        #torch.nn.Transformer(encoder_layer=layer, num_layers=num_layers)

    def forward(self, feats, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_is_causal=True, **kwargs):
        """
        feats - (Time, Batch, num_feats)
        tgt - (Time, Batch)
        src_key_padding_mask - (Batch, Time)
        tgt_key_padding_mask - (Batch, Time)
        """
        memory, lens = self.forward_encoder(feats, src_key_padding_mask=src_key_padding_mask, return_lens=True)
        memory_key_padding_mask = torch.arange(memory.shape[0], device=memory.device) >= lens[:, None]
        logits = self.forward_decoder(tgt, memory, memory_key_padding_mask=memory_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return logits  # (SeqLen, Batch, Classes)

    def forward_encoder(self, feats, src_key_padding_mask=None, return_lens=False, **kwargs):
        src_embs = self.input_ff(feats)
        # (S, B, E)
        src_embs = self.positional_encoding(src_embs)
        src_embs = src_embs.permute(1, 0, 2)
        # (B, S, E)
        lengths = feats.shape[0] - src_key_padding_mask.sum(dim=-1)
        # (B,)
        memory, lens = self.encoder(src_embs, lengths=lengths)
        # (B, S, E)
        memory = memory.permute(1, 0, 2)
        if return_lens:
            return memory, lens
        return memory

    def forward_decoder(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        tgt_embs = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embs = self.positional_encoding(tgt_embs)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[0], device=tgt.device)
        #print(tgt_embs.shape, memory.shape)
        embs = self.decoder(tgt_embs,
                            memory,
                            tgt_mask=tgt_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask)
        return self.head(embs)
