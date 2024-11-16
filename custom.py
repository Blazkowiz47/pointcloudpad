import os
from multiprocessing import Pool
from typing import List

import numpy as np


hyperparams = [
    #     {"optimizer": "ADAM", "lr": 0.001, "att_heads": 32},
    #     {"optimizer": "ADAM", "lr": 0.001, "att_heads": 16},
    {"optimizer": "ADAM", "lr": 0.001, "att_heads": 8},
    #         {"optimizer": "ADAM", "lr": 0.001, "att_heads": 4},
    #     {"optimizer": "ADAM", "lr": 0.001, "att_heads": 2},
    #     {"optimizer": "ADAM", "lr": 0.001, "att_heads": 1},
]


num_points = 1024
# dmodes = ["l", "n"]
# total_layers = [
#     [(i, i) for i in range(3, 6)],
#     [(i, j) for i in range(3, 6) for j in range(3, 6)],
# ]
dmodes = ["c"]
total_layers = [[(i, j) for i in range(2, 7) for j in range(1, 8)]]
total_layers = [[(3, 5)]]

train_iphones = ["iPhone11"]
train_iphones = ["iPhone11", "iPhone12"]
train_till_attack = 2
kernels = [1]


def train():
    args: List[str] = []
    for kernel in kernels:  # [1]:
        for iphone in train_iphones:
            for i in range(1, train_till_attack):
                attack = f"Attack_{i}"
                attack = "*"
                #             for p in hyperparams:
                #                 args.append(
                #                     f'python main.py  --model=DGCNN --iphone={iphone} --attack={attack} --epochs=10 --num_points={num_points} --emb_dims=1024 --k=20 --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]}'  # noqa: E501
                #                 )
                #             for p in hyperparams:
                #                 args.append(
                #                     f'python main.py  --model=AttentionDGCNN --epochs=10 --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                #                 )
                #             for p in hyperparams:
                #                 args.append(
                #                     f'python main.py  --model=PointNet --epochs=20 --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                #                 )
                #             for p in hyperparams:
                #                 args.append(
                #                     f'python main.py  --model=PointAttentionNet --epochs=20 --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                #                 )
                for dmode, layers in zip(dmodes, total_layers):
                    for ll, lr in layers:
                        for p in hyperparams:
                            args.append(
                                f'python main.py --kernel={kernel} --dmode={dmode} --layersleft={ll} --layersright={lr}  --model=DualDGCNN --epochs=40 --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                            )
                #             for p in hyperparams:
                #                 args.append(
                #                     f'python main.py  --model=ModDGCNN --epochs=20 --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                #                 )
        #             for p in hyperparams:
        #                 args.append(
        #                     f'python main.py  --model=DualDGCNNSA --epochs=100 --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
        #                 )
        #             for p in hyperparams:
        #                 args.append(
        #                     f'python main.py  --model=DualDGCNN7 --epochs=100 --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
        #                 )
        #             for p in hyperparams:
        #                 args.append(
        #                     f'python main.py  --model=DualDGCNNCA --epochs=100 --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
        #                 )

    with Pool(1) as p:
        p.map(os.system, args)


rdir = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/Intra"

attacks = [f"Attack_{i}" for i in range(1, 2)]
mattacks = [f"Attack_{i}" for i in range(9, 10)]
# ["Display-Attack", "Print-Attack"]


def test():
    args: List[str] = []
    for kernel in kernels:  # [1]:
        for iphone in train_iphones:
            for attack in attacks:
                attack = "*"
                for miphone in ["iPhone11", "iPhone12"]:
                    for mattack in mattacks:
                        #                     for p in hyperparams:
                        #                         args.append(
                        #                             f'python main.py --mode=test --model=DGCNN --miphone={miphone} --mattack={mattack} --iphone={iphone} --attack={attack} --epochs=10 --num_points={num_points} --emb_dims=1024 --k=20 --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]}'  # noqa: E501
                        #                         )
                        #                     for p in hyperparams:
                        #                         args.append(
                        #                             f'python main.py --mode=test --model=AttentionDGCNN --epochs=10 --miphone={miphone} --mattack={mattack} --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                        #                         )
                        #                     for p in hyperparams:
                        #                         args.append(
                        #                             f'python main.py --mode=test --model=PointNet --epochs=20 --miphone={miphone} --mattack={mattack} --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                        #                         )
                        #                     for p in hyperparams:
                        #                         args.append(
                        #                             f'python main.py --mode=test --model=PointAttentionNet --epochs=20 --miphone={miphone} --mattack={mattack} --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                        #                         )
                        for dmode, layers in zip(dmodes, total_layers):
                            for ll, lr in layers:
                                for p in hyperparams:
                                    args.append(
                                        f'python main.py --rdir="{rdir}" --kernel={kernel} --dmode={dmode} --layersleft={ll} --layersright={lr} --mode=test --model=DualDGCNN --epochs=20 --miphone={miphone} --mattack={mattack} --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                                    )
                        #                     for p in hyperparams:
                        #                         args.append(
                        #                             f'python main.py --mode=test --model=ModDGCNN --epochs=20 --miphone={miphone} --mattack={mattack} --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
                        #                         )
        #                     for p in hyperparams:
        #                         args.append(
        #                             f'python main.py --mode=test --model=DualDGCNN7 --epochs=20 --miphone={miphone} --mattack={mattack} --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
        #                         )
        #                     for p in hyperparams:
        #                         args.append(
        #                             f'python main.py --mode=test --model=DualDGCNNSA --epochs=20 --miphone={miphone} --mattack={mattack} --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
        #                         )
        #                     for p in hyperparams:
        #                         args.append(
        #                             f'python main.py --mode=test --model=DualDGCNNCA --epochs=20 --miphone={miphone} --mattack={mattack} --iphone={iphone} --attack={attack} --num_points={num_points} --emb_dims=1024 --k=20 --optimizer={p["optimizer"]} --lr={p["lr"]} --att_heads={p["att_heads"]} --momentum=0.9 --dropout=0.5 --dump_file=True --dry_run=False'  # noqa: E501
        #                         )

    with Pool(8) as p:
        p.map(os.system, args)


# train()
test()
