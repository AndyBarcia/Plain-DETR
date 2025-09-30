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

        return tgt

    def forward(
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
        if self.norm_type == 'pre_norm':
            tgt = self.forward_pre(tgt, query_pos, reference_points, src, src_pos_embed, src_spatial_shapes, src_padding_mask, self_attn_mask)
        elif self.norm_type == 'post_norm':
            tgt = self.forward_post(tgt, query_pos, reference_points, src, src_pos_embed, src_spatial_shapes, src_padding_mask, self_attn_mask)
        else:
            raise NotImplementedError
        return tgt


class GlobalDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        return_intermediate=False,
        look_forward_twice=False,
        use_checkpoint=False,
        d_model=256,
        norm_type="post_norm",
        reparam=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.bbox_embed = None
        self.class_embed = None
        self.norm = None
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice
        self.use_checkpoint = use_checkpoint
        self.d_model = d_model
        self.norm_type = norm_type
        self.reparam = reparam

        if self.norm_type == 'pre_norm':
            self.norm = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, src_padding_mask=None):
        if self.norm_type == 'pre_norm':
            tgt = self.norm(tgt)

        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    query_pos,
                    reference_points,
                    src,
                    src_spatial_shapes,
                    src_padding_mask,
                    None,
                )
            else:
                output = layer(
                    output,
                    query_pos,
                    reference_points,
                    src,
                    src_spatial_shapes,
                    src_padding_mask,
                    None,
                )
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

            if self.look_forward_twice and lid < self.num_layers - 1:
                # look forward twice
                if self.use_checkpoint:
                    output = checkpoint.checkpoint(
                        layer,
                        output,
                        query_pos,
                        reference_points,
                        src,
                        src_spatial_shapes,
                        src_padding_mask,
                        None,
                    )
                else:
                    output = layer(
                        output,
                        query_pos,
                        reference_points,
                        src,
                        src_spatial_shapes,
                        src_padding_mask,
                        None,
                    )
                if self.return_intermediate:
                    intermediate.append(output)
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output.unsqueeze(0), reference_points.unsqueeze(0)


def build_global_rpb_decoder(args):
    layer = GlobalDecoderLayer(
        d_model=args.hidden_dim,
        d_ffn=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        n_heads=args.nheads,
        norm_type=args.norm_type,
        rpe_hidden_dim=args.rpe_hidden_dim,
        rpe_type=args.decoder_rpe_type,
        feature_stride=args.feature_stride,
        reparam=args.reparam,
    )
    decoder = GlobalDecoder(
        layer,
        args.dec_layers,
        return_intermediate=args.return_intermediate_dec,
        look_forward_twice=args.look_forward_twice,
        use_checkpoint=args.use_checkpoint,
        d_model=args.hidden_dim,
        norm_type=args.norm_type,
        reparam=args.reparam,
    )
    return decoder
