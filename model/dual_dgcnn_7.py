import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import get_graph_feature


class DualDGCNN7(nn.Module):
    def __init__(self, args):
        super(DualDGCNN7, self).__init__()
        self.args = args
        self.k = args.k

        self.blk1 = EBlock(self.k, 3, 8 // 2, 1)
        self.blk2 = EBlock(self.k, 8 // 2, 16 // 2, args.att_heads)
        self.blk3 = EBlock(self.k, 16 // 2, 32 // 2, args.att_heads)
        self.blk4 = EBlock(self.k, 32 // 2, 64 // 2, args.att_heads)
        self.blk5 = EBlock(self.k, 64 // 2, 128 // 2, args.att_heads)
        self.blk6 = EBlock(self.k, 128 // 2, 256 // 2, args.att_heads)
        self.blk7 = EBlock(self.k, 256 // 2, 256 // 2, args.att_heads)

        self.blk8 = EBlock(self.k, 3, 8 // 2, 1)
        self.blk9 = EBlock(self.k, 8 // 2, 16 // 2, args.att_heads)
        self.blk10 = EBlock(self.k, 16 // 2, 32 // 2, args.att_heads)
        self.blk11 = EBlock(self.k, 32 // 2, 64 // 2, args.att_heads)
        self.blk12 = EBlock(self.k, 64 // 2, 128 // 2, args.att_heads)
        self.blk13 = EBlock(self.k, 128 // 2, 256 // 2, args.att_heads)
        self.blk14 = EBlock(self.k, 256 // 2, 256 // 2, args.att_heads)

        self.conv1_l1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv1_l2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Sequential(
            nn.Conv1d(760, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, args.number_classes)

    def forward(self, x):
        batch_size = x.size(0)

        x1 = self.blk1(x)
        x2 = self.blk2(x1)
        x3 = self.blk3(x2)
        x4 = self.blk4(x3)
        x5 = self.blk5(x4)
        x6 = self.blk6(x5)
        x7 = self.blk7(x6)

        x8 = self.blk8(x)
        x9 = self.blk9(x8)
        x10 = self.blk10(x9)
        x11 = self.blk11(x10)
        x12 = self.blk12(x11)
        x13 = self.blk13(x12)
        x14 = self.blk14(x13)

        x = torch.cat(
            (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14), dim=1
        )

        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class EBlock(nn.Module):
    def __init__(self, k, indim, outdim, att_heads) -> None:
        super(EBlock, self).__init__()
        self.attn = nn.MultiheadAttention(outdim, att_heads)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * indim, outdim, kernel_size=1, bias=False),
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
