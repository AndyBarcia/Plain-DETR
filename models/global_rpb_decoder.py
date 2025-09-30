# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import numpy as np
from timm.models.layers import trunc_normal_

from util.misc import inverse_sigmoid, _get_clones, _get_activation_fn
from util.box_ops import box_xyxy_to_cxcywh, delta2bbox

from models.pos_mlp_bias.functions.box_rpb import PosMLPAttention, PosGaussianAttention


class GlobalCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rpe_hidden_dim=512,
        rpe_type='linear',
        feature_stride=16,
        reparam=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.rpe_type = rpe_type
        self.feature_stride = feature_stride
        self.reparam = reparam

        self.is_standard = rpe_type not in ['pos_mlp', 'pos_gaussian']
        self.cross_attn = None

        if self.is_standard:
            self.cpb_mlp1 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
            self.cpb_mlp2 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
            self.softmax = nn.Softmax(dim=-1)
        else:
            if rpe_type == 'pos_mlp':
                self.cross_attn = PosMLPAttention(
                    dim=dim,
                    num_heads=num_heads,
                    hidden_dim=rpe_hidden_dim,
                    dropout=attn_drop,
                    implementation='cuda'
                )
            elif rpe_type == 'pos_gaussian':
                self.cross_attn = PosGaussianAttention(
                    dim=dim,
                    num_heads=num_heads,
                    k_dim=dim,
                    dropout=attn_drop,
                    implementation='cuda'
                )

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, out_dim, bias=False))
        return cpb_mlp

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        raw_query,
        query_pos,
        reference_points,
        raw_src,
        src_pos_embed,
        input_spatial_shapes,
        input_padding_mask=None,
    ):
        assert input_spatial_shapes.size(0) == 1, 'This is designed for single-scale decoder.'
        h, w = input_spatial_shapes[0]
        B, Nq, C = raw_query.shape
        HW = h * w

        if reference_points.dim() == 4:
            reference_points = reference_points.squeeze(2)

        if self.is_standard:
            # Standard RPE computation
            # Add positional embeddings
            if query_pos.dim() == 2:
                query = raw_query + query_pos[None, :, :]
            else:
                query = raw_query + query_pos

            if src_pos_embed.dim() == 2:
                k_input_flatten = raw_src + src_pos_embed[None, :, :]
            else:
                k_input_flatten = raw_src + src_pos_embed

            v_input_flatten = raw_src

            stride = self.feature_stride

            ref_pts = torch.cat([
                reference_points[:, :, :2] - reference_points[:, :, 2:] / 2,
                reference_points[:, :, :2] + reference_points[:, :, 2:] / 2,
            ], dim=-1)  # B, nQ, 4
            if not self.reparam:
                ref_pts[..., 0::2] *= (w * stride)
                ref_pts[..., 1::2] *= (h * stride)
            pos_x = torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=w.device)[None, None, :, None] * stride  # 1, 1, w, 1
            pos_y = torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=h.device)[None, None, :, None] * stride  # 1, 1, h, 1

            if self.rpe_type == 'abs_log8':
                delta_x = ref_pts[..., 0::2] - pos_x  # B, nQ, w, 2
                delta_y = ref_pts[..., 1::2] - pos_y  # B, nQ, h, 2
                delta_x = torch.sign(delta_x) * torch.log2(torch.abs(delta_x) + 1.0) / np.log2(8)
                delta_y = torch.sign(delta_y) * torch.log2(torch.abs(delta_y) + 1.0) / np.log2(8)
            elif self.rpe_type == 'linear':
                delta_x = ref_pts[..., 0::2] - pos_x  # B, nQ, w, 2
                delta_y = ref_pts[..., 1::2] - pos_y  # B, nQ, h, 2
            else:
                raise NotImplementedError

            rpe_x, rpe_y = self.cpb_mlp1(delta_x), self.cpb_mlp2(delta_y)  # B, nQ, w/h, nheads
            rpe = (rpe_x[:, :, None] + rpe_y[:, :, :, None]).flatten(2, 3) # B, nQ, h, w, nheads ->  B, nQ, h*w, nheads
            rpe = rpe.permute(0, 3, 1, 2)

            k = self.k(k_input_flatten).reshape(B, HW, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(v_input_flatten).reshape(B, HW, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = self.q(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q * self.scale

            attn = q @ k.transpose(-2, -1)
            attn += rpe
            if input_padding_mask is not None:
                attn += input_padding_mask[:, None, None] * -100

            fmin, fmax = torch.finfo(attn.dtype).min, torch.finfo(attn.dtype).max
            torch.clip_(attn, min=fmin, max=fmax)

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = attn @ v

            x = x.transpose(1, 2).reshape(B, Nq, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        else:
            # PosMLP or PosGaussian mode
            # Add positional embeddings
            if query_pos.dim() == 2:
                queries = raw_query + query_pos[None, :, :]
            else:
                queries = raw_query + query_pos

            memory = raw_src.view(B, h, w, C)
            if src_pos_embed.dim() == 2:
                src_pos_embed_exp = src_pos_embed[None, :, :].view(B, h, w, C)
            else:
                src_pos_embed_exp = src_pos_embed.view(1, h, w, C)
            memory = memory + src_pos_embed_exp

            # Prepare attn_mask
            if input_padding_mask is not None:
                # Assume input_padding_mask is (B, HW), bool or float; convert to bool for masked_fill
                if input_padding_mask.dtype != torch.bool:
                    input_padding_mask = input_padding_mask > 0.5
                attn_mask = input_padding_mask[:, None, None, :].expand(B, self.num_heads, Nq, HW)
            else:
                attn_mask = None

            out, _ = self.cross_attn.forward_inner(
                queries,
                memory,
                reference_points,
                query_pos_emb=None,
                memory_pos_emb=None,
                attn_mask=attn_mask,
                return_attn_logits=False
            )
            return out


class GlobalDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
        norm_type='post_norm',
        rpe_hidden_dim=512,
        rpe_type='box_norm',
        feature_stride=16,
        reparam=False,
    ):
        super().__init__()

        self.norm_type = norm_type

        # global cross attention
        self.cross_attn = GlobalCrossAttention(d_model, n_heads, rpe_hidden_dim=rpe_hidden_dim,
                                               rpe_type=rpe_type, feature_stride=feature_stride,
                                               reparam=reparam)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        tgt2 = self.norm2(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt2.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)

        # global cross attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn(
            tgt2,
            query_pos,
            reference_points,
            src,
            src_pos_embed,
            src_spatial_shapes,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

    def forward_post(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            tgt,
            query_pos,
            reference_points,
            src,
            src_pos_embed,
            src_spatial_shapes,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)