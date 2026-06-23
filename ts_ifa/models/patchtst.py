"""PatchTST forecasting model adapted from the original local implementation."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.dims != (1, 2):
            raise ValueError("Transpose only supports dims=(1, 2) in this package")
        out = rearrange(x, "batch time dim -> batch dim time")
        return out.contiguous() if self.contiguous else out


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = repeat(x[:, 0, :], "batch dim -> batch pad dim", pad=pad)
        end = repeat(x[:, -1, :], "batch dim -> batch pad dim", pad=pad)
        x, _ = pack([front, x, end], "batch * dim")
        x = self.avg(rearrange(x, "batch time dim -> batch dim time"))
        return rearrange(x, "batch dim time -> batch time dim")


class series_decomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = rearrange(torch.arange(0, q_len), "q -> q 1")
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    del verbose
    x = 0.5 if exponential else 1
    for _ in range(100):
        cpe = (
            2
            * (rearrange(torch.linspace(0, 1, q_len), "q -> q 1") ** x)
            * (rearrange(torch.linspace(0, 1, d_model), "d -> 1 d") ** x)
            - 1
        )
        if abs(cpe.mean()) <= eps:
            break
        if cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = 2 * (
        rearrange(torch.linspace(0, 1, q_len), "q -> q 1") ** (0.5 if exponential else 1)
    ) - 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe is None:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "zeros":
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "normal" or pe == "gauss":
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == "exp2d":
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == "sincos":
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: "
            "'gauss'=='normal', 'zeros', 'zero', uniform', 'lin1d', 'exp1d', "
            "'lin2d', 'exp2d', 'sincos', None.)"
        )
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class PatchTST_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        d_model=128,
        n_heads=16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout=0,
        padding_patch=None,
        pretrain_head: bool = False,
        head_type="flatten",
        individual=False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
            **kwargs,
        )

        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == "flatten":
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def forward(self, z):
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = rearrange(z, "batch vars patch_num patch_len -> batch vars patch_len patch_num")

        z = self.backbone(z)
        return self.head(z)

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            for _ in range(self.n_vars):
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = rearrange(x[:, i, :, :], "batch dim patch -> batch (dim patch)")
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = rearrange(x_out, "vars batch horizon -> batch vars horizon")
        else:
            x = rearrange(x, "batch vars dim patch -> batch vars (dim patch)")
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        patch_num,
        patch_len,
        max_seq_len=1024,
        n_layers=3,
        d_model=128,
        n_heads=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        act="gelu",
        store_attn=False,
        key_padding_mask="auto",
        padding_var=None,
        attn_mask=None,
        res_attention=True,
        pre_norm=False,
        pe="zeros",
        learn_pe=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        del c_in, max_seq_len, key_padding_mask, padding_var, attn_mask, verbose, kwargs

        self.patch_num = patch_num
        self.patch_len = patch_len

        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TSTEncoder(
            q_len,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def forward(self, x) -> Tensor:
        n_vars = x.shape[1]
        x = rearrange(x, "batch vars patch_len patch_num -> batch vars patch_num patch_len")
        x = self.W_P(x)

        u = rearrange(x, "batch vars patch dim -> (batch vars) patch dim")
        u = self.dropout(u + self.W_pos)

        z = self.encoder(u)
        z = rearrange(z, "(batch vars) patch dim -> batch vars patch dim", vars=n_vars)
        return rearrange(z, "batch vars patch dim -> batch vars dim patch")


class TSTEncoder(nn.Module):
    def __init__(
        self,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=None,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        res_attention=False,
        n_layers=1,
        pre_norm=False,
        store_attn=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len,
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for _ in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        for mod in self.layers:
            output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=256,
        store_attn=False,
        norm="BatchNorm",
        attn_dropout=0,
        dropout=0.0,
        bias=True,
        activation="gelu",
        res_attention=False,
        pre_norm=False,
    ):
        super().__init__()
        del q_len
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        return src


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        lsa=False,
    ):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )

        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        q_s = rearrange(
            self.W_Q(Q),
            "batch seq (heads dim) -> batch heads seq dim",
            heads=self.n_heads,
        )
        k_s = rearrange(
            self.W_K(K),
            "batch seq (heads dim) -> batch heads dim seq",
            heads=self.n_heads,
        )
        v_s = rearrange(
            self.W_V(V),
            "batch seq (heads dim) -> batch heads seq dim",
            heads=self.n_heads,
        )

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        output = rearrange(output, "batch heads seq dim -> batch seq (heads dim)")
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        attn_scores = torch.matmul(q, k) * self.scale

        if prev is not None:
            attn_scores = attn_scores + prev

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:
            mask = rearrange(key_padding_mask, "batch seq -> batch 1 1 seq")
            attn_scores.masked_fill_(mask, -np.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class PatchTST(nn.Module):
    def __init__(
        self,
        lags: int | None = None,
        dim: int = 1,
        horizon: int | None = None,
        *,
        context_window: int | None = None,
        target_window: int | None = None,
        name: str | None = None,
        max_seq_len: int | None = None,
        d_k: int | None = None,
        d_v: int | None = None,
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool | str = "auto",
        padding_var: int | None = None,
        attn_mask: Tensor | None = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        verbose: bool = False,
        c_in: int = 1,
        n_layers: int = 3,
        n_heads: int = 16,
        d_model: int = 128,
        d_ff: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        individual: int | bool = 0,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str | None = "end",
        **kwargs,
    ):
        super().__init__()
        del name

        context_window = context_window if context_window is not None else lags
        target_window = target_window if target_window is not None else horizon
        context_window = 24 * 7 if context_window is None else int(context_window)
        target_window = 24 if target_window is None else int(target_window)
        max_seq_len = context_window if max_seq_len is None else int(max_seq_len)

        self.lags = context_window
        self.dim = int(dim)
        self.horizon = target_window

        self.model = PatchTST_backbone(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            patch_len=patch_len,
            stride=stride,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=individual,
            verbose=verbose,
            **kwargs,
        )

    def forward(self, x: Tensor, covariates=None, **kwargs) -> Tensor:
        del covariates, kwargs
        if x.shape[1] != self.dim or x.shape[-1] != self.lags:
            raise ValueError(
                f"expected input shape (batch, {self.dim}, {self.lags}), got {tuple(x.shape)}"
            )
        return self.model(x)

    @torch.no_grad()
    def representation(self, x: Tensor, *, pool: bool = True, **kwargs) -> Tensor:
        del kwargs
        if x.shape[1] != self.dim or x.shape[-1] != self.lags:
            raise ValueError(
                f"expected input shape (batch, {self.dim}, {self.lags}), got {tuple(x.shape)}"
            )
        z = x
        if self.model.padding_patch == "end":
            z = self.model.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.model.patch_len, step=self.model.stride)
        z = rearrange(z, "batch vars patch_num patch_len -> batch vars patch_len patch_num")
        encoded = self.model.backbone(z)
        if pool:
            return encoded.mean(dim=(-1, -2))
        return rearrange(encoded, "batch ... -> batch (...)")
