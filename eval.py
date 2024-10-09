import os

from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd


hyperparams = [
    #     {"optimizer": "ADAM", "lr": 0.001, "att_heads": 32},
    #     {"optimizer": "ADAM", "lr": 0.001, "att_heads": 16},
    {"optimizer": "ADAM", "lr": 0.001, "att_heads": 8},
    #     {"optimizer": "ADAM", "lr": 0.001, "att_heads": 4},
    #     {"optimizer": "ADAM", "lr": 0.001, "att_heads": 2},
    #     {"optimizer": "ADAM", "lr": 0.001, "att_heads": 1},
]

num_points = 1024


def get_eer(model, heads, iphone, attack, miphone, mattack) -> float:
    from common_metrics import eer

    rdir = "/cluster/nbl-users/Shreyas-Sushrut-Raghu/PAD-Features/DifferentDual/"
    genfile = os.path.join(
        rdir,
        model + "_" + str(heads),
        f"trained_on_{iphone}_{attack}/gen_{miphone}_{mattack}.txt",
    )
    atkfile = os.path.join(
        rdir,
        model + "_" + str(heads),
        f"trained_on_{iphone}_{attack}/atk_{miphone}_{mattack}.txt",
    )
    genscores = np.loadtxt(genfile)
    atkscores = np.loadtxt(atkfile)
    try:
        return round(eer(genscores, atkscores), 3)
    except:
        print(genscores.shape)
        print(atkscores.shape)
        return 100


dmodes = ["l", "n"]
total_layers = [
    [(i, i) for i in range(3, 6)],
    [(i, j) for i in range(3, 6) for j in range(3, 6)],
]

dmodes = ["c"]
total_layers = [[(i, j) for i in range(3, 6) for j in range(3, 6)]]

dmodes = ["c"]
# total_layers = [[(i, j) for i in range(3, 6) for j in range(3, 6)]]
total_layers = [[(3, i) for i in range(1, 8)]]
total_layers = [[(i, j) for i in range(2, 7) for j in range(1, 8)]]
total_layers = [[(3, 5)]]

train_till_attack = 2


def eval():
    for iphone in ["iPhone12"]:
        for i in range(1, train_till_attack):
            attack = f"Attack_{i}"
            print("Trained on:", iphone, attack)
            dfraw = {
                "model": [],
                "iphone": [],
                "attack": [],
                "eer": [],
            }

            #             for m in ["DualDGCNN"]:  # ,"DualDGCNN7", "DualDGCNNSA", "DualDGCNNCA"
            for kernel in [1, 3, 5, 7]:
                for dmode, layers in zip(dmodes, total_layers):
                    for ll, lr in layers:
                        for miphone in ["iPhone11", "iPhone12"]:
                            for i in range(1, 7):
                                mattack = f"Attack_{i}"

                                for p in hyperparams:
                                    res = str(
                                        get_eer(
                                            f"Ablation_DualDGCNN_DifferentDual_{ll}_{lr}_{dmode}_k{kernel}",
                                            p["att_heads"],
                                            iphone,
                                            attack,
                                            miphone,
                                            mattack,
                                        )
                                    )
                                    dfraw["model"].append(
                                        f"Abl_Dual_DD_{ll}_{lr}_{dmode}_k{kernel}_{p['att_heads']}"
                                    )
                                    dfraw["eer"].append(res)
                                    dfraw["iphone"].append(miphone)
                                    dfraw["attack"].append(mattack)

            models = list(set(dfraw["model"]))

            df = pd.DataFrame(dfraw)
            df_raw_reformat = {"iphone": [], "attack": [], **{k: [] for k in models}}
            for iphone in ["iPhone12"]:
                for i in range(6, 7):
                    attack = f"Attack_{i}"
                    temp = df.query(f'iphone=="{iphone}" and attack=="{attack}"')
                    for m in sorted(temp["model"].tolist()):
                        x = temp.query(f'model=="{m}"')
                        df_raw_reformat[m].append(x.iloc[0]["eer"])
                    df_raw_reformat["iphone"].append(iphone)
                    df_raw_reformat["attack"].append(attack)
            df_reformat = pd.DataFrame(df_raw_reformat)
            x = df_reformat.query("attack=='Attack_6' and iphone=='iPhone12'")

            pd.options.display.max_columns = None
            pd.options.display.max_rows = None
            print(x)


eval()
