from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import get_graph_feature


class DualDGCNN(nn.Module):
    def __init__(self, args):
        super(DualDGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.dmode = args.dmode
        self.kernel = args.kernel

        if self.dmode == "l":
            self.build_l(args)
        elif self.dmode == "n":
            self.build_n(args)
        else:
            self.build_c(args)

        self.kernel = args.kernel

        self.conv1_l1 = nn.Conv2d(
            6,
            64,
            kernel_size=self.kernel,
            padding=(self.kernel - 1) // 2,
            stride=1,
            bias=False,
        )
        self.conv1_l2 = nn.BatchNorm2d(64)

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, args.number_classes)

    def cblocks(self, dims, emb_dims, layers) -> Tuple[List[nn.Module], int]:
        out_layers: List[nn.Module] = [
            nn.Sequential(
                nn.Conv1d(
                    dims,
                    emb_dims,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(emb_dims),
                nn.LeakyReLU(negative_slope=0.2),
            )
        ]
        for _ in range(layers - 1):
            out_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        emb_dims,
                        emb_dims,
                        kernel_size=self.kernel,
                        padding=(self.kernel - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(emb_dims),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
        return out_layers, layers * emb_dims

    def blocks(self, layers, heads) -> Tuple[List[nn.Module], Tuple[int, int]]:
        if layers == 1:
            return [
                EBlock(self.k, 3, 256 // 2, heads, self.kernel),
            ], (0, 128)
        elif layers == 2:
            return [
                EBlock(self.k, 3, 64 // 2, heads, self.kernel),
                EBlock(self.k, 64 // 2, 256 // 2, heads, self.kernel),
            ], (32, 128)
        elif layers == 3:
            return [
                EBlock(self.k, 3, 64 // 2, heads, self.kernel),
                EBlock(self.k, 64 // 2, 128 // 2, heads, self.kernel),
                EBlock(self.k, 128 // 2, 256 // 2, heads, self.kernel),
            ], (32 + 64, 128)
        elif layers == 4:
            return [
                EBlock(self.k, 3, 64 // 2, heads, self.kernel),
                EBlock(self.k, 64 // 2, 64 // 2, heads, self.kernel),
                EBlock(self.k, 64 // 2, 128 // 2, heads, self.kernel),
                EBlock(self.k, 128 // 2, 256 // 2, heads, self.kernel),
            ], (32 + 32 + 64, 128)

        elif layers == 5:
            return [
                EBlock(self.k, 3, 32 // 2, heads, self.kernel),
                EBlock(self.k, 32 // 2, 64 // 2, heads, self.kernel),
                EBlock(self.k, 64 // 2, 64 // 2, heads, self.kernel),
                EBlock(self.k, 64 // 2, 128 // 2, heads, self.kernel),
                EBlock(self.k, 128 // 2, 256 // 2, heads, self.kernel),
            ], (16 + 32 + 32 + 64, 128)
        elif layers == 6:
            return [
                EBlock(self.k, 3, 32 // 2, heads, self.kernel),
                EBlock(self.k, 32 // 2, 64 // 2, heads, self.kernel),
                EBlock(self.k, 64 // 2, 64 // 2, heads, self.kernel),
                EBlock(self.k, 64 // 2, 128 // 2, heads, self.kernel),
                EBlock(self.k, 128 // 2, 128 // 2, heads, self.kernel),
                EBlock(self.k, 128 // 2, 256 // 2, heads, self.kernel),
            ], (16 + 32 + 32 + 64 + 64, 128)
        raise NotImplementedError()

    def build_l(self, args):
        blocks, dims = self.blocks(args.layersleft, args.att_heads)
        self.stream = nn.Sequential(*blocks)

        self.conv5 = nn.Sequential(
            nn.Conv1d(dims, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def build_c(self, args):
        blocks, (diml, dimr) = self.blocks(args.layersleft, args.att_heads)
        self.stream = nn.Sequential(*blocks)

        blocks, dl = self.cblocks(diml, args.emb_dims, args.layersright)
        self.sl = nn.Sequential(*blocks)
        blocks, dr = self.cblocks(dimr, args.emb_dims, args.layersright)
        self.sr = nn.Sequential(*blocks)

        self.conv9 = nn.Sequential(
            nn.Conv1d(dl + dr, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def build_n(self, args):
        blocks, diml = self.blocks(args.layersleft, args.att_heads)
        self.stream_left = nn.Sequential(*blocks)

        blocks, dimr = self.blocks(args.layersright, args.att_heads)
        self.stream_right = nn.Sequential(*blocks)

        self.conv5 = nn.Sequential(
            nn.Conv1d(sum(diml) + sum(dimr), args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward_c(self, x):
        skips = []
        for layer in self.stream:
            x = layer(x)
            skips.append(x)

        xl = torch.cat(skips[:-1], dim=1)
        xr = skips[-1]

        skips = []
        x = xl
        for layer in self.sl:
            x = layer(x)
            skips.append(x)

        x = xr
        for layer in self.sr:
            x = layer(x)
            skips.append(x)

        x = torch.cat(skips, dim=1)
        x = self.conv9(x)
        return x

    def forward_l(self, x):
        skips = []
        for layer in self.stream:
            x = layer(x)
            skips.append(x)

        x = torch.cat(skips, dim=1)

        x = self.conv5(x)
        return x

    def forward_n(self, x):
        skips = []
        xo = x
        for layer in self.stream_left:
            x = layer(x)
            skips.append(x)
        x = xo
        for layer in self.stream_right:
            x = layer(x)
            skips.append(x)

        x = torch.cat(skips, dim=1)

        x = self.conv5(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        if self.dmode == "l":
            x = self.forward_l(x)
        elif self.dmode == "c" or "a" in self.dmode:
            x = self.forward_c(x)
        else:
            x = self.forward_n(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x2 = x
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x1 = x
        x = self.linear3(x)
        return x, x1, x2


class EBlock(nn.Module):
    def __init__(self, k, indim, outdim, att_heads, kernel) -> None:
        super(EBlock, self).__init__()
        self.kernel = kernel
        self.attn = nn.MultiheadAttention(outdim, att_heads)
        self.conv = nn.Sequential(
            nn.Conv2d(
                2 * indim,
                outdim,
                kernel_size=self.kernel,
                padding=(self.kernel - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(outdim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.k = k

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)  # [32, 6, 1024, 40]
        x = self.conv(x)  # [32, 64, 1024, 40]
        x = x.max(dim=-1, keepdim=False)[0]  # [32, 64, 1024]

        residual = x
        x1_T = x.transpose(1, 2)  # [32, 1024, 64]
        x1_att, _ = self.attn(x1_T, x1_T, x1_T)  # [32, 1024, 64]
        x = x1_att.transpose(1, 2)  # [32, 64, 1024]
        del x1_T, x1_att, _
        x += residual
        return x
