import torch
from torch import nn
from torch.nn import functional as F
from non_local import NONLocalBlock2D
from rn import RN_B, RN_L
import numpy as np

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

def totxt(tensor, name):
    tmp = np.array(tensor.squeeze(0).cpu())
    np.savetxt(f'{name}.txt', tmp)

class Quantize_vqvae(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim) #[1024, 64] | []
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock_vqvae(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder_vqvae(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock_vqvae(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder_vqvae(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock_vqvae(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channels=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder_vqvae(in_channel, channel, n_res_block, n_res_channel, stride=4)
        # self.enc_m = Encoder_vqvae(channel, channel, n_res_block, n_res_channel, stride=2)
        self.enc_t = Encoder_vqvae(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize_vqvae(embed_dim, n_embed)
        self.dec_t = Decoder_vqvae(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize_vqvae(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder_vqvae(
            embed_dim + embed_dim,
            out_channels,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        # self.dec_s = Decoder_vqvae(
        #     in_channel=embed_dim, out_channel=out_channels, channel=channel, n_res_block=n_res_block, n_res_channel=n_res_channel, stride=4,
        # )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input) #[N, 128, 64, 64]
        enc_t = self.enc_t(enc_b) #[N, 128, 32, 32]

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1) #[N, 64, 32, 32] -> [N, 32, 32, 64]
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2) #[N, 64, 32, 32]
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t) #[N, 64, 64, 64]
        enc_b = torch.cat([dec_t, enc_b], 1) #[N, 64+128, 64, 64]

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1) #[N, 64, 64, 64]
        quant_b, diff_b, id_b = self.quantize_b(quant_b) #[N, 64, 64, 64]
        quant_b = quant_b.permute(0, 3, 1, 2) #[N, 64, 64, 64]
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t) #[N, 64, 64, 64]
        quant = torch.cat([upsample_t, quant_b], 1)  #[N, 128, 64, 64]
        dec = self.dec(quant) #[N, 3, 256, 256]
        # dec_segment = self.dec_s(upsample_t) #[N, 9, 256, 256]

        return dec

class Quantize_2(nn.Module):
    def __init__(self, dim, n_embed, memory_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = memory_embed
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input, fix=1):
        flatten = input.reshape(-1, self.dim) #[1024, 64] | []
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])

        quantize = self.embed_code(embed_ind)

        if fix == 0:
            if self.training:
                self.cluster_size.data.mul_(self.decay).add_(
                    1 - self.decay, embed_onehot.sum(0)
                )
                embed_sum = flatten.transpose(0, 1) @ embed_onehot
                self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
                self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

def Norm_layer(norm_type, dim_in, ifb=None):
    if norm_type == 'bn':
        return nn.BatchNorm2d(dim_in)
    elif norm_type == 'in':
        return nn.InstanceNorm2d(dim_in, track_running_stats=False)
    elif norm_type == 'rn':
        if ifb == 'e':
            return RN_B(dim_in)
        elif ifb == 'd':
            return RN_L(dim_in)
        elif ifb == 'r':
            return RN_L(dim_in, threshold=0.8)
    else:
        return None

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, norm_type, use_spectral_norm=True):
        super().__init__()

        if norm_type == 'rn':
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                spectral_norm(nn.Conv2d(in_channel, channel, 3, padding=1, bias=not use_spectral_norm), use_spectral_norm),
                Norm_layer(norm_type, channel, 'r'),
                nn.ReLU(inplace=True),
                spectral_norm(nn.Conv2d(channel, in_channel, 1, bias=not use_spectral_norm), use_spectral_norm),
                Norm_layer(norm_type, in_channel, 'r')
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, channel, 3, padding=1),
                Norm_layer(norm_type, channel, 'r'),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, in_channel, 1),
                Norm_layer(norm_type, in_channel, 'r')
            )

    def forward(self, input):
        out = self.conv(input)
        out = input + out

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, norm_type, nl):
        super().__init__()

        if stride == 4:
            self.conv1 = nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1)
            self.norm1 = Norm_layer(norm_type, channel // 2, 'e')
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1)
            self.norm2 = Norm_layer(norm_type, channel, 'e')
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(channel, channel, 3, padding=1)
            self.norm3 = Norm_layer(norm_type, channel, 'e')
        elif stride == 2:
            self.conv1 = nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1)
            self.norm1 = Norm_layer(norm_type, channel // 2, 'e')
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(channel // 2, channel, 3, padding=1)
            self.norm2 = Norm_layer(norm_type, channel, 'e')

        blocks = []
        if nl == 1:
            blocks.append(NONLocalBlock2D(channel))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel, norm_type))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, mask, s, norm):
        if s == 4:
            x = self.conv1(input)
            x = self.norm1(x, mask) if norm == 'rn' else self.norm1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.norm2(x, mask) if norm == 'rn' else self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.norm3(x, mask) if norm == 'rn' else self.norm3(x)
        elif s == 2:
            x = self.conv1(input)
            x = self.norm1(x, mask) if norm == 'rn' else self.norm1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.norm2(x, mask) if norm == 'rn' else self.norm2(x)
        return self.blocks(x)

class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, norm_type):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1),
                  Norm_layer(norm_type, channel, 'd'),
                  ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel, norm_type))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    Norm_layer(norm_type, channel // 2, 'd'),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                    Norm_layer(norm_type, out_channel, 'd'),
                ]
            )

        elif stride == 2:
            blocks.extend([
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1),
                Norm_layer(norm_type, out_channel, 'd'),
                ]
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Channel_Fuse(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, norm_type, use_spectral_norm=True):
        super().__init__()

        if norm_type == 'rn':
            self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size, bias=not use_spectral_norm), use_spectral_norm),
            Norm_layer(norm_type, dim_out, 'd'),
            nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size),
                Norm_layer(norm_type, dim_out, 'd'),
                nn.ReLU(inplace=True),
            )

    def forward(self, input):
        out = self.conv(input)
        return out
        

class Stage_2(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        embed_b=None,
        embed_t=None
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, 6, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, 6, n_res_channel, stride=2)
        self.quantize_conv_t = Channel_Fuse(channel, embed_dim, 1)
        self.fuse_top_feat = Channel_Fuse(channel, embed_dim, 1)
        self.quantize_t = Quantize_2(embed_dim, n_embed, memory_embed=embed_t)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = Channel_Fuse(embed_dim + channel, embed_dim, 1)
        self.fuse_bottom_feat = Channel_Fuse(channel, embed_dim, 1)
        self.quantize_b = Quantize_2(embed_dim, n_embed, memory_embed=embed_b)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim, 9, channel, n_res_block, n_res_channel, stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input) #[128, 64, 64]
        enc_t = self.enc_t(enc_b) #[128, 32, 32]

        enc_t_bofore_vq = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1) #[64, 32, 32] -> [32, 32, 64]
        quant_t, diff_t, id_t = self.quantize_t(enc_t_bofore_vq)
        quant_t = quant_t.permute(0, 3, 1, 2) #[64, 32, 32]
        diff_t = diff_t.unsqueeze(0)
        quant_t = torch.cat([quant_t, enc_t_bofore_vq.permute(0, 3, 1, 2)], 1) #[128, 32, 32]
        quant_t = self.fuse_top_feat(quant_t) #[64, 32, 32]

        dec_t = self.dec_t(quant_t) #[64, 64, 64]
        enc_b = torch.cat([dec_t, enc_b], 1) #[64+128, 64, 64]

        enc_b_before_vq = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1) #[64, 64, 64]
        quant_b, diff_b, id_b = self.quantize_b(enc_b_before_vq)
        quant_b = quant_b.permute(0, 3, 1, 2) #[64, 64, 64]
        diff_b = diff_b.unsqueeze(0)
        quant_b = torch.cat([quant_b, enc_b_before_vq.permute(0, 3, 1, 2)], 1) #[128, 64, 64]
        quant_b = self.fuse_bottom_feat(quant_b) #[64, 64, 64]

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t) #[64, 64, 64]
        quant = torch.cat([upsample_t, quant_b], 1)  #[256, 64, 64]
        dec = self.dec(quant)

        return dec

class Stage_2_Inner_Constraint(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channels=9,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        norm_layer=None,
        embed_b=None,
        embed_t=None,
        nl=1,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, 8, n_res_channel, stride=4, norm_type=norm_layer, nl=nl)
        self.enc_t = Encoder(channel, channel, 6, n_res_channel, stride=2, norm_type=norm_layer, nl=nl)
        self.quantize_conv_t = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
        self.fuse_top_feat = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
        self.quantize_t = Quantize_2(embed_dim, n_embed, memory_embed=embed_t)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2, norm_type=norm_layer)
        self.quantize_conv_b = Channel_Fuse(embed_dim + channel, embed_dim, 1, norm_type=norm_layer)
        self.fuse_bottom_feat = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
        self.quantize_b = Quantize_2(embed_dim, n_embed, memory_embed=embed_b)
        self.upsample_t = nn.Sequential(nn.ConvTranspose2d( embed_dim, embed_dim, 4, stride=2, padding=1),Norm_layer(norm_layer, embed_dim, 'd'),nn.ReLU(inplace=True))
        self.dec = Decoder(embed_dim + embed_dim, out_channels, channel, n_res_block, n_res_channel, stride=4, norm_type=norm_layer)
        self.dec_coarse_b = Decoder(embed_dim + embed_dim, out_channels, channel, n_res_block, n_res_channel, stride=4, norm_type=norm_layer)

    def forward(self, input, mask, norm, fix_memory):
        quant_t, quant_b, coarse_b_feat, diff, _, _ = self.encode(input, mask, norm, fix_memory)
        dec = self.decode(quant_t, quant_b)
        coarse_b = self.dec_coarse_b(coarse_b_feat)
        # dec, coarse_b = (torch.tanh(dec)+1)/2, (torch.tanh(coarse_b)+1)/2

        return dec, diff, coarse_b

    def encode(self, input, mask, norm, fix_memory):
        enc_b = self.enc_b(input, mask, 4, norm) #[128, 64, 64]
        coarse_b_feat = enc_b
        enc_t = self.enc_t(enc_b, mask, 2, norm) #[128, 32, 32]
        enc_t_bofore_vq = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1) #[64, 32, 32] -> [32, 32, 64]
        quant_t, diff_t, id_t = self.quantize_t(enc_t_bofore_vq, fix=fix_memory)
        quant_t = quant_t.permute(0, 3, 1, 2) #[64, 32, 32]
        diff_t = diff_t.unsqueeze(0)
        quant_t = torch.cat([quant_t, enc_t_bofore_vq.permute(0, 3, 1, 2)], 1) #[128, 32, 32]
        quant_t = self.fuse_top_feat(quant_t) #[64, 32, 32]

        dec_t = self.dec_t(quant_t) #[64, 64, 64]
        enc_b = torch.cat([dec_t, enc_b], 1) #[64+128, 64, 64]
        enc_b_before_vq = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1) #[64, 64, 64]
        quant_b, diff_b, id_b = self.quantize_b(enc_b_before_vq, fix=fix_memory)
        quant_b = quant_b.permute(0, 3, 1, 2) #[64, 64, 64]
        diff_b = diff_b.unsqueeze(0)
        quant_b = torch.cat([quant_b, enc_b_before_vq.permute(0, 3, 1, 2)], 1) #[128, 64, 64]
        quant_b = self.fuse_bottom_feat(quant_b) #[64, 64, 64]

        return quant_t, quant_b, coarse_b_feat, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t) #[64, 64, 64]
        quant = torch.cat([upsample_t, quant_b], 1)  #[128, 64, 64]
        dec = self.dec(quant)

        return dec

# class Stage_2_ind_pred(nn.Module):
#     def __init__(
#         self,
#         in_channel=3,
#         channel=128,
#         n_res_block=2,
#         n_res_channel=32,
#         embed_dim=64,
#         n_embed=512,
#         decay=0.99,
#         norm_layer=None,
#         embed_b=None,
#         embed_t=None,
#         nl=1,
#     ):
#         super().__init__()

#         self.enc_b = Encoder(in_channel, channel, 8, n_res_channel, stride=4, norm_type=norm_layer, nl=nl)
#         self.enc_t = Encoder(channel, channel, 6, n_res_channel, stride=2, norm_type=norm_layer, nl=nl)
#         self.quantize_conv_t = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
#         self.fuse_top_feat = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
#         self.quantize_t = Quantize_3(embed_dim, n_embed, memory_embed=embed_t)
#         self.pred_t = nn.Sequential(nn.Conv2d(embed_dim, n_embed, 1), Norm_layer(norm_layer, n_embed, 'd'), nn.Sigmoid())
#         self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2, norm_type=norm_layer)
#         self.quantize_conv_b = Channel_Fuse(embed_dim + channel, embed_dim, 1, norm_type=norm_layer)
#         self.fuse_bottom_feat = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
#         self.quantize_b = Quantize_3(embed_dim, n_embed, memory_embed=embed_b)
#         self.pred_b = nn.Sequential(nn.Conv2d(embed_dim, n_embed, 1), Norm_layer(norm_layer, n_embed, 'd'), nn.Sigmoid())
#         self.upsample_t = nn.Sequential(nn.ConvTranspose2d( embed_dim, embed_dim, 4, stride=2, padding=1),Norm_layer(norm_layer, embed_dim, 'd'),nn.ReLU(inplace=True))
#         self.dec = Decoder(embed_dim + embed_dim, 9, channel, n_res_block, n_res_channel, stride=4, norm_type=norm_layer)
#         self.dec_coarse_b = Decoder(embed_dim + embed_dim, 9, channel, n_res_block, n_res_channel, stride=4, norm_type=norm_layer)

#     def forward(self, input, mask, norm, fix_memory):
#         quant_t, quant_b, coarse_b_feat, diff, _, _ = self.encode(input, mask, norm, fix_memory)
#         dec = self.decode(quant_t, quant_b)
#         coarse_b = self.dec_coarse_b(coarse_b_feat)
#         dec, coarse_b = (torch.tanh(dec)+1)/2, (torch.tanh(coarse_b)+1)/2

#         return dec, diff, coarse_b

#     def encode(self, input, mask, norm, fix_memory):
#         enc_b = self.enc_b(input, mask, 4, norm) #[128, 64, 64]
#         coarse_b_feat = enc_b
#         enc_t = self.enc_t(enc_b, mask, 2, norm) #[128, 32, 32]
#         enc_t_bofore_vq = self.quantize_conv_t(enc_t) #[64, 32, 32]
#         ind_t = self.pred_t(enc_t_bofore_vq) #[512, 32, 32]
#         quant_t, diff_t, id_t = self.quantize_t(enc_t_bofore_vq.permute(0, 2, 3, 1), ind_t, fix=fix_memory)
#         quant_t = quant_t.permute(0, 3, 1, 2) #[64, 32, 32]
#         diff_t = diff_t.unsqueeze(0)
#         # quant_t = torch.cat([quant_t, enc_t_bofore_vq], 1) #[128, 32, 32]
#         # quant_t = self.fuse_top_feat(quant_t) #[64, 32, 32]

#         dec_t = self.dec_t(quant_t) #[64, 64, 64]
#         enc_b = torch.cat([dec_t, enc_b], 1) #[64+128, 64, 64]
#         enc_b_before_vq = self.quantize_conv_b(enc_b) #[64, 64, 64]
#         ind_b = self.pred_b(enc_b_before_vq) #[512, 64, 64]
#         quant_b, diff_b, id_b = self.quantize_b(enc_b_before_vq.permute(0, 2, 3, 1), ind_b, fix=fix_memory)
#         quant_b = quant_b.permute(0, 3, 1, 2) #[64, 64, 64]
#         diff_b = diff_b.unsqueeze(0)
#         # quant_b = torch.cat([quant_b, enc_b_before_vq], 1) #[128, 64, 64]
#         # quant_b = self.fuse_bottom_feat(quant_b) #[64, 64, 64]

#         return quant_t, quant_b, coarse_b_feat, diff_t + diff_b, id_t, id_b

#     def decode(self, quant_t, quant_b):
#         upsample_t = self.upsample_t(quant_t) #[64, 64, 64]
#         quant = torch.cat([upsample_t, quant_b], 1)  #[128, 64, 64]
#         dec = self.dec(quant)

#         return dec

class Stage_3_Inner_Constraint(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        norm_layer=None,
        embed_b=None,
        embed_t=None,
        nl=1,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, 8, n_res_channel, stride=4, norm_type=norm_layer, nl=nl)
        self.enc_t = Encoder(channel, channel, 6, n_res_channel, stride=2, norm_type=norm_layer, nl=nl)
        self.quantize_conv_t = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
        self.fuse_top_feat = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
        self.quantize_t = Quantize_2(embed_dim, n_embed, memory_embed=embed_t)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2, norm_type=norm_layer)
        self.quantize_conv_b = Channel_Fuse(embed_dim + channel, embed_dim, 1, norm_type=norm_layer)
        self.fuse_bottom_feat = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
        self.quantize_b = Quantize_2(embed_dim, n_embed, memory_embed=embed_b)
        self.upsample_t = nn.Sequential(nn.ConvTranspose2d( embed_dim, embed_dim, 4, stride=2, padding=1),Norm_layer(norm_layer, embed_dim, 'd'),nn.ReLU(inplace=True))
        self.dec = Decoder(embed_dim + embed_dim, 3, channel, n_res_block, n_res_channel, stride=4, norm_type=norm_layer)
        self.dec_coarse_b = Decoder(embed_dim + embed_dim, 3, channel, n_res_block, n_res_channel, stride=4, norm_type=norm_layer)

    def forward(self, input, mask, norm, fix_memory):
        quant_t, quant_b, coarse_b_feat, diff, _, _ = self.encode(input, mask, norm, fix_memory)
        dec = self.decode(quant_t, quant_b)
        coarse_b = self.dec_coarse_b(coarse_b_feat)
        dec, coarse_b = (torch.tanh(dec)+1)/2, (torch.tanh(coarse_b)+1)/2

        return dec, diff, coarse_b

    def encode(self, input, mask, norm, fix_memory):
        enc_b = self.enc_b(input, mask, 4, norm) #[128, 64, 64]
        coarse_b_feat = enc_b
        enc_t = self.enc_t(enc_b, mask, 2, norm) #[128, 32, 32]
        enc_t_bofore_vq = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1) #[64, 32, 32] -> [32, 32, 64]
        quant_t, diff_t, id_t = self.quantize_t(enc_t_bofore_vq, fix=fix_memory)
        quant_t = quant_t.permute(0, 3, 1, 2) #[64, 32, 32]
        diff_t = diff_t.unsqueeze(0)
        quant_t = torch.cat([quant_t, enc_t_bofore_vq.permute(0, 3, 1, 2)], 1) #[128, 32, 32]
        quant_t = self.fuse_top_feat(quant_t) #[64, 32, 32]

        dec_t = self.dec_t(quant_t) #[64, 64, 64]
        enc_b = torch.cat([dec_t, enc_b], 1) #[64+128, 64, 64]
        enc_b_before_vq = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1) #[64, 64, 64]
        quant_b, diff_b, id_b = self.quantize_b(enc_b_before_vq, fix=fix_memory)
        quant_b = quant_b.permute(0, 3, 1, 2) #[64, 64, 64]
        diff_b = diff_b.unsqueeze(0)
        quant_b = torch.cat([quant_b, enc_b_before_vq.permute(0, 3, 1, 2)], 1) #[128, 64, 64]
        quant_b = self.fuse_bottom_feat(quant_b) #[64, 64, 64]

        return quant_t, quant_b, coarse_b_feat, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t) #[64, 64, 64]
        quant = torch.cat([upsample_t, quant_b], 1)  #[128, 64, 64]
        dec = self.dec(quant)

        return dec

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Quantize_3(nn.Module):
    def __init__(self, dim, n_embed, memory_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = memory_embed
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input, ind, fix=1):
        flatten = input.reshape(-1, self.dim) #[1024, 64] | []
        embed_ind = torch.argmax(ind, dim=1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if fix == 0:
            if self.training:
                self.cluster_size.data.mul_(self.decay).add_(
                    1 - self.decay, embed_onehot.sum(0)
                )
                embed_sum = flatten.transpose(0, 1) @ embed_onehot
                self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
                self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

# class Stage_3_ind_pred(nn.Module):
#     def __init__(
#         self,
#         in_channel=3,
#         channel=128,
#         n_res_block=2,
#         n_res_channel=32,
#         embed_dim=64,
#         n_embed=512,
#         decay=0.99,
#         norm_layer=None,
#         embed_b=None,
#         embed_t=None,
#         nl=1,
#     ):
#         super().__init__()

#         self.enc_b = Encoder(in_channel, channel, 8, n_res_channel, stride=4, norm_type=norm_layer, nl=nl)
#         self.enc_t = Encoder(channel, channel, 6, n_res_channel, stride=2, norm_type=norm_layer, nl=nl)
#         self.quantize_conv_t = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
#         self.fuse_top_feat = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
#         self.quantize_t = Quantize_3(embed_dim, n_embed, memory_embed=embed_t)
#         self.pred_t = nn.Sequential(nn.Conv2d(embed_dim, n_embed, 1), Norm_layer(norm_layer, n_embed, 'd'), nn.Sigmoid())
#         self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2, norm_type=norm_layer)
#         self.quantize_conv_b = Channel_Fuse(embed_dim + channel, embed_dim, 1, norm_type=norm_layer)
#         self.fuse_bottom_feat = Channel_Fuse(channel, embed_dim, 1, norm_type=norm_layer)
#         self.quantize_b = Quantize_3(embed_dim, n_embed, memory_embed=embed_b)
#         self.pred_b = nn.Sequential(nn.Conv2d(embed_dim, n_embed, 1), Norm_layer(norm_layer, n_embed, 'd'), nn.Sigmoid())
#         self.upsample_t = nn.Sequential(nn.ConvTranspose2d( embed_dim, embed_dim, 4, stride=2, padding=1),Norm_layer(norm_layer, embed_dim, 'd'),nn.ReLU(inplace=True))
#         self.dec = Decoder(embed_dim + embed_dim, 3, channel, n_res_block, n_res_channel, stride=4, norm_type=norm_layer)
#         self.dec_coarse_b = Decoder(embed_dim + embed_dim, 3, channel, n_res_block, n_res_channel, stride=4, norm_type=norm_layer)

#     def forward(self, input, mask, norm, fix_memory):
#         quant_t, quant_b, coarse_b_feat, diff, _, _ = self.encode(input, mask, norm, fix_memory)
#         dec = self.decode(quant_t, quant_b)
#         coarse_b = self.dec_coarse_b(coarse_b_feat)
#         dec, coarse_b = (torch.tanh(dec)+1)/2, (torch.tanh(coarse_b)+1)/2

#         return dec, diff, coarse_b

#     def encode(self, input, mask, norm, fix_memory):
#         enc_b = self.enc_b(input, mask, 4, norm) #[128, 64, 64]
#         coarse_b_feat = enc_b
#         enc_t = self.enc_t(enc_b, mask, 2, norm) #[128, 32, 32]
#         enc_t_bofore_vq = self.quantize_conv_t(enc_t) #[64, 32, 32]
#         ind_t = self.pred_t(enc_t_bofore_vq) #[512, 32, 32]
#         quant_t, diff_t, id_t = self.quantize_t(enc_t_bofore_vq.permute(0, 2, 3, 1), ind_t, fix=fix_memory)
#         quant_t = quant_t.permute(0, 3, 1, 2) #[64, 32, 32]
#         diff_t = diff_t.unsqueeze(0)
#         # quant_t = torch.cat([quant_t, enc_t_bofore_vq], 1) #[128, 32, 32]
#         # quant_t = self.fuse_top_feat(quant_t) #[64, 32, 32]

#         dec_t = self.dec_t(quant_t) #[64, 64, 64]
#         enc_b = torch.cat([dec_t, enc_b], 1) #[64+128, 64, 64]
#         enc_b_before_vq = self.quantize_conv_b(enc_b) #[64, 64, 64]
#         ind_b = self.pred_b(enc_b_before_vq) #[512, 64, 64]
#         quant_b, diff_b, id_b = self.quantize_b(enc_b_before_vq.permute(0, 2, 3, 1), ind_b, fix=fix_memory)
#         quant_b = quant_b.permute(0, 3, 1, 2) #[64, 64, 64]
#         diff_b = diff_b.unsqueeze(0)
#         # quant_b = torch.cat([quant_b, enc_b_before_vq], 1) #[128, 64, 64]
#         # quant_b = self.fuse_bottom_feat(quant_b) #[64, 64, 64]

#         return quant_t, quant_b, coarse_b_feat, diff_t + diff_b, id_t, id_b

#     def decode(self, quant_t, quant_b):
#         upsample_t = self.upsample_t(quant_t) #[64, 64, 64]
#         quant = torch.cat([upsample_t, quant_b], 1)  #[128, 64, 64]
#         dec = self.dec(quant)

#         return dec

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
