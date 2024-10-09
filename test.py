import argparse

import torch

from model.dual_dgcnn import DualDGCNN
from model.dual_dgcnn_sattn import DualDGCNNSA
from model.dual_dgcnn_cattn import DualDGCNNCA

m = DualDGCNN(
    argparse.Namespace(
        k=20,
        num_points=1024,
        emb_dims=1024,
        att_heads=8,
        momentum=0.9,
        dropout=0.5,
        number_classes=2,
        dmode="c",
        layersleft=3,
        layersright=4,
        kernel=1,
    )
).cuda()

x = torch.rand((32, 1024, 3)).cuda()
x = x.permute(0, 2, 1)

m(x)
