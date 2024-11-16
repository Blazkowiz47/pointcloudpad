#!/usr/bin/env python
import argparse
from typing import Any, List, Tuple
import random
from dataset.custom import Custom
from utils.params import Params

import time
import numpy as np
import torch
import torch.nn as nn
from utils.utility import calculate_loss

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from multiprocessing import Pool
import os

# import model
from model.attention_dgcnn import AttentionDGCNN
from model.dgcnn import DGCNN
from model.point_attention_net import PointAttentionNet
from model.point_net import PointNet
from model.modified_dgcnn import ModDGCNN
from model.dual_dgcnn import DualDGCNN
from model.dual_dgcnn_7 import DualDGCNN7
from model.dual_dgcnn_sattn import DualDGCNNSA
from model.dual_dgcnn_cattn import DualDGCNNCA


import sklearn.metrics as metrics


def train(args: Params):
    print("Training:", args.iphone, args.attack)
    train_loader = DataLoader(
        args.dataset_loader(
            args.iphone,
            args.attack,
            partition="Train",
            num_points=args.num_points,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    validation_loader = DataLoader(
        args.dataset_loader(
            args.iphone,
            args.attack,
            partition="Vali",
            num_points=args.num_points,
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
    )
    device = args.device
    model = args.model(args).to(args.device)
    args.log(str(model), False)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.optimizer == "SGD":
        print(f"{str(args.model)} use SGD")
        opt = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4
        )
    else:
        print(f"{str(args.model)} use Adam")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = calculate_loss

    if args.last_checkpoint() != "":
        exit()
        model.load_state_dict(torch.load(args.last_checkpoint()))

    global_best_loss, global_best_acc, global_best_avg_acc = 0, 0, 0
    for epoch in range(args.epochs):
        epoch_results = []
        ts = time.time()

        def train_batch():
            ####################
            # Train
            ####################
            train_loss = 0.0
            count = 0.0
            model.train()
            train_pred = []
            train_true = []
            for data, label in train_loader:
                batch_size = data.size()[0]
                if batch_size == 1:
                    data = torch.concat([data, data], axis=0)
                    label = torch.concat([label, label], axis=0)
                batch_size = data.size()[0]
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                opt.zero_grad()
                logits = model(data)
                if logits.shape[1] != 2:
                    print(str(args.model), args.iphone, args.attack, args.att_heads)
                    exit()

                loss = criterion(logits, label)
                loss.backward()
                opt.step()
                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())

                if args.dry_ryn:
                    break

            scheduler.step()
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            return train_loss * 1.0 / count, train_true, train_pred

        train_loss, train_true, train_pred = train_batch()
        if args.dry_ryn:
            break

        ####################
        # Validation
        ####################
        with torch.no_grad():
            val_loss = 0.0
            count = 0.0
            model.eval()
            val_pred = []
            val_true = []

            # best_val_loss, best_val_acc, best_val_avg_acc = 0, 0, 0
            for data, label in validation_loader:
                batch_size = data.size()[0]
                if batch_size == 1:
                    data = torch.concat([data, data], axis=0)
                    label = torch.concat([label, label], axis=0)
                batch_size = data.size()[0]
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                val_loss += loss.item() * batch_size
                val_true.append(label.cpu().numpy())
                val_pred.append(preds.detach().cpu().numpy())

            val_true = np.concatenate(val_true)
            val_pred = np.concatenate(val_pred)
            val_acc = metrics.accuracy_score(val_true, val_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(val_true, val_pred)

            args.csv(
                epoch,
                train_loss,
                metrics.accuracy_score(train_true, train_pred),
                metrics.balanced_accuracy_score(train_true, train_pred),
                val_loss * 1.0 / count,
                val_acc,
                avg_per_class_acc,
                time.time() - ts,
            )

            checkpoint_path = args.checkpoint_path()
            torch.save(model.state_dict(), checkpoint_path)
            if avg_per_class_acc > global_best_avg_acc:
                global_best_loss, global_best_acc, global_best_avg_acc = (
                    val_loss * 1.0 / count,
                    val_acc,
                    avg_per_class_acc,
                )
                torch.save(model.state_dict(), args.best_checkpoint())
            print(checkpoint_path)

    args.print_summary(global_best_loss, global_best_acc, global_best_avg_acc)


def test_driver(miphone, mattack, params: Params, state_dict=None):
    test(miphone, mattack, params, state_dict)


def test(iphone, attack, params: Params, state_dict=None):
    print("=====================================================================")
    print("Testing:", iphone, attack)
    args = params
    test_loader = DataLoader(
        args.dataset_loader(
            iphone,
            attack,
            partition="Test",
            num_points=args.num_points,
            rdir=args.rdir,
        ),
        batch_size=args.test_batch_size,
        drop_last=False,
        num_workers=4,
        shuffle=True,
    )
    rdir = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/PAD-Features/DifferentDual/"
    #     if os.path.isfile(
    #         f"{rdir}/{params.model_name or params.model.__name__}_{params.att_heads}/trained_on_{args.iphone}_{args.attack}/atk_{iphone}_{attack}.txt"
    #     ) and os.path.isfile(
    #         f"{rdir}/{params.model_name or params.model.__name__}_{params.att_heads}/trained_on_{args.iphone}_{args.attack}/gen_{iphone}_{attack}.txt"
    #     ):
    #         return

    device = args.device
    model = params.model(params).to(params.device)

    if state_dict != None:
        model.load_state_dict(torch.load(state_dict))
    else:
        checkpoint = params.best_checkpoint()
        print(checkpoint)
        model.load_state_dict(torch.load(checkpoint))

    genuine_scores = []
    morph_scores = []

    with torch.no_grad():
        sfmx = torch.nn.Softmax(dim=1).cuda()
        model.eval()
        test_acc = 0.0
        count = 0.0
        test_true = []
        test_pred = []
        for data, label in test_loader:
            batch_size = data.size()[0]
            if batch_size == 1:
                data = torch.concat([data, data], axis=0)
                label = torch.concat([label, label], axis=0)
            batch_size = data.size()[0]
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = model(data)
            logits = sfmx(logits)
            if logits.shape[1] != 2:
                args.log(logits.shape)
            label = label.detach().cpu().numpy()
            preds = logits.argmax(dim=1)
            logits = logits.detach().cpu().numpy()
            for x, y in zip(logits, label):
                if y:
                    genuine_scores.append(x[1])
                else:
                    morph_scores.append(x[1])

            test_true.append(label)
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        if state_dict == None:
            outstr = "TEST:: test acc: %.6f, test avg acc: %.6f" % (
                test_acc,
                avg_per_class_acc,
            )
            args.log(
                "===================================================================="
            )
            args.log(outstr)

    os.makedirs(
        f"{rdir}/{params.model_name or params.model.__name__}_{params.att_heads}/trained_on_{args.iphone}_{args.attack}",
        exist_ok=True,
    )

    np.savetxt(
        f"{rdir}/{params.model_name or params.model.__name__}_{params.att_heads}/trained_on_{args.iphone}_{args.attack}/gen_{iphone}_{attack}.txt",
        np.array(genuine_scores),
    )
    np.savetxt(
        f"{rdir}/{params.model_name or params.model.__name__}_{params.att_heads}/trained_on_{args.iphone}_{args.attack}/atk_{iphone}_{attack}.txt",
        np.array(morph_scores),
    )


def get_model(model):
    if model == "DGCNN":
        return DGCNN
    if model == "AttentionDGCNN":
        return AttentionDGCNN
    if model == "PointNet":
        return PointNet
    if model == "PointAttentionNet":
        return PointAttentionNet
    if model == "DualDGCNN":
        return DualDGCNN
    if model == "ModDGCNN":
        return ModDGCNN
    if model == "DualDGCNN7":
        return DualDGCNN7
    if model == "DualDGCNNSA":
        return DualDGCNNSA
    if model == "DualDGCNNCA":
        return DualDGCNNCA


def driver(args: argparse.Namespace) -> None:
    params = Params(
        mode=args.mode,
        model=get_model(args.model),
        iphone=args.iphone,
        attack=args.attack,
        epochs=args.epochs,
        num_points=args.num_points,
        emb_dims=args.emb_dims,
        k=args.k,
        optimizer=args.optimizer,
        lr=args.lr,
        att_heads=args.att_heads,
        momentum=args.momentum,
        dropout=args.dropout,
        dump_file=args.dump_file,
        dry_run=args.dry_run,
        model_name=f"Ablation_DualDGCNN_DifferentDual_{args.layersleft}_{args.layersright}_{args.dmode}_k{args.kernel}",
        rdir=args.rdir,
    )
    params.kernel = args.kernel
    params.dmode = args.dmode
    params.layersleft = args.layersleft
    params.layersright = args.layersright
    if args.mode == "test":
        test_driver(args.miphone, args.mattack, params)
    else:
        train(params)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="")
parser.add_argument("--model", type=str, default="DGCNN")
parser.add_argument("--miphone", type=str, default="")
parser.add_argument("--mattack", type=str, default="")
parser.add_argument("--iphone", type=str, default="iPhone12")
parser.add_argument("--attack", type=str, default="Attack_6")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_points", type=int, default=1024)
parser.add_argument("--emb_dims", type=int, default=256)
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--att_heads", type=int, default=4)
parser.add_argument("--momentum", type=float, default=0.1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--dump_file", type=bool, default=True)
parser.add_argument("--dry_run", type=bool, default=False)
parser.add_argument("--dmode", type=str, default="n")
parser.add_argument("--kernel", type=int, default=1)
parser.add_argument("--layersleft", type=int, default=4)
parser.add_argument("--layersright", type=int, default=4)
parser.add_argument("--rdir", type=str, default="/home/ubuntu/work/Intra/")


def set_seeds(seed: int = 2024):
    """
    Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = parser.parse_args()
    set_seeds()
    driver(args)
