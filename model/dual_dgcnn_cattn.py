import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import get_graph_feature


class DualDGCNNCA(nn.Module):
    def __init__(self, args):
        super(DualDGCNNCA, self).__init__()
        self.args = args
        self.k = args.k

        self.blk1 = EBlockCA(self.k, 3, 64 // 2, args.att_heads)
        self.blk2 = EBlockCA(self.k, 64 // 2, 64 // 2, args.att_heads)
        self.blk3 = EBlockCA(self.k, 64 // 2, 128 // 2, args.att_heads)
        self.blk4 = EBlockCA(self.k, 128 // 2, 256 // 2, args.att_heads)

        self.conv1_l1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv1_l2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Sequential(
            nn.Conv1d(256 * 2, args.emb_dims, kernel_size=1, bias=False),
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

        x1, x5 = self.blk1(x)
        x2, x6 = self.blk2(x1)
        x3, x7 = self.blk3(x2)
        x4, x8 = self.blk4(x3)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

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


class EBlockCA(nn.Module):
    def __init__(self, k, indim, outdim, att_heads) -> None:
        super(EBlockCA, self).__init__()
        self.attnl = nn.MultiheadAttention(outdim, att_heads)
        self.convl = nn.Sequential(
            nn.Conv2d(2 * indim, outdim, kernel_size=1, bias=False),
            nn.BatchNorm2d(outdim),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.attnr = nn.MultiheadAttention(outdim, att_heads)
        self.convr = nn.Sequential(
            nn.Conv2d(2 * indim, outdim, kernel_size=1, bias=False),
            nn.BatchNorm2d(outdim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.k = k

    def forward(self, x):
        xl = get_graph_feature(x, k=self.k)  # [32, 6, 1024, 40]
        xl = self.convl(xl)  # [32, 64, 1024, 40]
        xl = xl.max(dim=-1, keepdim=False)[0]  # [32, 64, 1024]

        xr = get_graph_feature(x, k=self.k)  # [32, 6, 1024, 40]
        xr = self.convr(xr)  # [32, 64, 1024, 40]
        xr = xr.max(dim=-1, keepdim=False)[0]  # [32, 64, 1024]

        residual = xl
        x1_T = xl.transpose(1, 2)  # [32, 1024, 64]

        residual = xr
        x2_T = xr.transpose(1, 2)  # [32, 1024, 64]

        x1_att, _ = self.attnl(x2_T, x1_T, x1_T)  # [32, 1024, 64]
        del _
        xl = x1_att.transpose(1, 2)  # [32, 64, 1024]
        xl += residual

        x2_att, _ = self.attnr(x1_T, x2_T, x2_T)  # [32, 1024, 64]
        del _
        xr = x2_att.transpose(1, 2)  # [32, 64, 1024]
        xr += residual

        del x1_T, x1_att
        del x2_T, x2_att

        return xl, xr
