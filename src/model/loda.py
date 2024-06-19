"""
based on
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
"""

from collections import OrderedDict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Mlp, VisionTransformer

from .patch_embed import PatchEmbed


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cosine


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, image_token):
        B, N, C = image_token.shape
        kv = (
            self.kv(image_token)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        B, N, C = query.shape
        q = (
            self.q(query)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Learner(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        num_classes = cfg.model.learner_param.num_classes
        embed_dim = cfg.model.learner_param.embed_dim
        feature_channels = cfg.model.learner_param.feature_channels
        cnn_feature_num = cfg.model.learner_param.cnn_feature_num
        interaction_block_num = cfg.model.learner_param.interaction_block_num
        latent_dim = cfg.model.learner_param.latent_dim
        grid_size = cfg.model.learner_param.grid_size
        cross_attn_num_heads = cfg.model.learner_param.cross_attn_num_heads

        # hyper net
        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            feature_channels[i],
                            latent_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                        nn.GELU(),
                        nn.Conv2d(
                            latent_dim,
                            latent_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.AdaptiveAvgPool2d(grid_size),
                    ]
                )
                for i in range(cnn_feature_num)
            ]
        )
        self.down_proj = nn.ModuleList(
            [
                Mlp(
                    in_features=embed_dim,
                    hidden_features=latent_dim,
                    out_features=latent_dim,
                )
                for _ in range(interaction_block_num)
            ]
        )
        self.cross_attn = nn.ModuleList(
            [
                CrossAttention(dim=latent_dim, num_heads=cross_attn_num_heads)
                for _ in range(interaction_block_num)
            ]
        )
        self.up_proj = nn.ModuleList(
            [
                Mlp(
                    in_features=latent_dim,
                    hidden_features=embed_dim,
                    out_features=embed_dim,
                )
                for _ in range(interaction_block_num)
            ]
        )
        self.scale_factor = nn.Parameter(
            torch.randn(interaction_block_num, embed_dim) * 0.02
        )

        # new head
        self.head = NormedLinear(embed_dim, num_classes)

        self._init_parameters()

    def _init_parameters(self):
        trunc_normal_(self.scale_factor, std=0.02)

    def forward(self, x):
        return self.head(x)


class LoDa(VisionTransformer):
    def __init__(
        self,
        cfg=None,
        embed_layer=PatchEmbed,
        basic_state_dict=None,
        *argv,
        **karg,
    ):
        # Recreate ViT
        super().__init__(
            embed_layer=embed_layer,
            *argv,
            **karg,
            **(cfg.model.vit_param),
        )

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.learner = Learner(cfg)
        self.dropout = nn.Dropout(cfg.model.hyper_vit.dropout_rate)
        self.head = nn.Identity()

        # feature_extraction model
        self.feature_model = timm.create_model(
            cfg.model.feature_model.name,
            pretrained=cfg.model.feature_model.load_timm_model,
            features_only=True,
            out_indices=cfg.model.feature_model.out_indices,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.learner.parameters():
            param.requires_grad = True

    def un_freeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_state_to_save(self):
        feature_model_state_dict = self.feature_model.state_dict()
        bn_buffer = OrderedDict()
        for key, value in feature_model_state_dict.items():
            if (
                "running_mean" in key
                or "running_var" in key
                or "num_batches_tracked" in key
            ):
                bn_buffer[key] = value

        state_dict_to_save = {
            "learner": self.learner.state_dict(),
            "bn_buffer": bn_buffer,
        }
        return state_dict_to_save

    def load_saved_state(self, saved_state_dict, strict=False):
        self.learner.load_state_dict(saved_state_dict["learner"], strict)
        self.feature_model.load_state_dict(saved_state_dict["bn_buffer"], strict)

    def forward_hyper_net(self, x):
        batch_size = x.shape[0]
        features_list = self.feature_model(x)

        cnn_token_list = []
        for i in range(len(features_list)):
            cnn_image_token = self.learner.conv[i](features_list[i])
            latent_dim = cnn_image_token.shape[1]
            cnn_image_token = cnn_image_token.permute(0, 2, 3, 1).reshape(
                batch_size, -1, latent_dim
            )
            cnn_token_list.append(cnn_image_token)

        return torch.cat(cnn_token_list, dim=1)

    def forward_features(self, x, cnn_tokens):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i in range(len(self.blocks)):
            x_down = self.learner.down_proj[i](x)
            x_down = x_down + self.learner.cross_attn[i](x_down, cnn_tokens)
            x_up = self.learner.up_proj[i](x_down)
            x = x + x_up * self.learner.scale_factor[i]
            x = self.blocks[i](x)

        x = self.norm(x)
        return x[:, 0, :]

    def forward(self, x):
        cnn_tokens = self.forward_hyper_net(x)
        x = self.forward_features(x, cnn_tokens)

        x = self.dropout(x)
        x = self.learner(x)
        x = self.head(x)
        return x
