import os

from multiprocessing import Pool
from typing import List

import numpy as np
from numpy.typing import NDArray
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


def eer(genuine, imposter, bins=10_001) -> float:
    genuine = np.squeeze(np.array(genuine))
    imposter = np.squeeze(np.array(imposter))
    far = np.ones(bins)
    frr = np.ones(bins)
    mi = np.min(imposter)
    mx = np.max(genuine)
    thresholds = np.linspace(mi, mx, bins)
    for id, threshold in enumerate(thresholds):
        fr = np.where(genuine <= threshold)[0].shape[0]
        fa = np.where(imposter >= threshold)[0].shape[0]
        frr[id] = fr * 100 / genuine.shape[0]
        far[id] = fa * 100 / imposter.shape[0]

    di = np.argmin(np.abs(far - frr))

    eer = (far[di] + frr[di]) / 2
    return eer


def get_eer(model, heads, iphone, attack, miphone, mattack) -> float:
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

attacks = [f"Attack_{i}" for i in range(1, 7)]
mattacks = ["Display-Attack", "Print-Attack"]
iphones = ["iPhone11", "iPhone12"]


def eval():
    for iphone in iphones:
        for attack in attacks:
            print("Trained on:", iphone, attack)
            dfraw = {
                "model": [],
                "iphone": [],
                "attack": [],
                "eer": [],
            }

            #             for m in ["DualDGCNN"]:  # ,"DualDGCNN7", "DualDGCNNSA", "DualDGCNNCA"
            for kernel in [1]:
                for dmode, layers in zip(dmodes, total_layers):
                    for ll, lr in layers:
                        for miphone in ["iPhone11", "iPhone12"]:
                            for mattack in mattacks:
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
                                    dfraw["tiphone"].append(iphone)
                                    dfraw["tattack"].append(attack)

            models = list(set(dfraw["model"]))

            df = pd.DataFrame(dfraw)
            df_raw_reformat = {"iphone": [], "attack": [], **{k: [] for k in models}}
            for iphone in iphones:
                for attack in mattacks:
                    temp = df.query(f'iphone=="{iphone}" and attack=="{attack}"')
                    for m in sorted(temp["model"].tolist()):
                        x = temp.query(f'model=="{m}"')
                        df_raw_reformat[m].append(x.iloc[0]["eer"])
                    df_raw_reformat["iphone"].append(iphone)
                    df_raw_reformat["attack"].append(attack)
            df_reformat = pd.DataFrame(df_raw_reformat)
            #             x = df_reformat.query("attack=='Attack_6' and iphone=='iPhone12'")

            pd.options.display.max_columns = None
            pd.options.display.max_rows = None
            print(df_reformat)


eval()
