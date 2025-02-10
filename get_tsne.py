import os
from typing import Tuple

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


iphones = ["iPhone11", "iPhone12"]
attacks = [f"Attack_{i}" for i in range(1, 9)]

FT1_ROOT = "<Root-Directory>"
FT2_ROOT = "<Root-Directory>"
LBL_ROOT = "<Root-Directory>"


def get_files(iphone, attack) -> Tuple[str, str, str]:
    return (
        os.path.join(FT1_ROOT, f"atk_{iphone}_{attack}.npy"),
        os.path.join(FT2_ROOT, f"atk_{iphone}_{attack}.npy"),
        os.path.join(LBL_ROOT, f"atk_{iphone}_{attack}.npy"),
    )


FEAT1 = None
FEAT2 = None
LABEL = None
IPHONE = []
ATTACK = []

for iphone in iphones:
    for attack in attacks:
        feat1_file, feat2_file, lbl_file = get_files(iphone, attack)
        feat1, feat2, lbl = np.load(feat1_file), np.load(feat2_file), np.load(lbl_file)
        if FEAT1 is None:
            FEAT1 = feat1
        else:
            FEAT1 = np.concatenate((FEAT1, feat1), axis=0)
        if FEAT2 is None:
            FEAT2 = feat2
        else:
            FEAT2 = np.concatenate((FEAT2, feat2), axis=0)
        if LABEL is None:
            LABEL = lbl
        else:
            LABEL = np.concatenate((LABEL, lbl), axis=0)

        IPHONE += [iphone] * feat1.shape[0]
        ATTACK += [attack] * feat1.shape[0]

if FEAT1 is None or FEAT2 is None or LABEL is None:
    raise ValueError()

IPHONE = np.array(IPHONE)
ATTACK = np.array(ATTACK)

print(
    FEAT1.shape,
    FEAT2.shape,
    LABEL.shape,
    IPHONE.shape,
    ATTACK.shape,
    np.logical_and(
        np.logical_and(LABEL == 0, ATTACK == "Attack_1"), IPHONE == "iPhone11"
    ).shape,
)

tsne = TSNE(n_components=2, init="random", perplexity=50, n_jobs=24)
pca = PCA(n_components=2, svd_solver="full")

FEAT1_EMB = pca.fit_transform(FEAT1)
FEAT2_EMB = pca.fit_transform(FEAT2)


def plot_features(
    features, labels, liphones, lattacks, fname: str, tiphone: str
) -> None:
    plt.figure(fname, figsize=(10, 10))
    total_patches = []
    n = 0
    colors = cm.rainbow(np.linspace(0, 1, len(attacks) + 1))
    for label in [0, 1]:
        if label:
            c = colors[n]
            feat = features[np.logical_and(LABEL == label, liphones == tiphone)]
            plt.scatter(
                feat[:, 0],
                feat[:, 1],
                color=c,
                marker="+",
                alpha=0.12,
            )
            n += 1
            total_patches.append(mpatches.Patch(color=c, label="Real data"))

        else:
            for iphone in iphones:
                if iphone != tiphone:
                    continue
                for attack in attacks:
                    c = colors[n]
                    feat = features[
                        np.logical_and(
                            np.logical_and(labels == label, lattacks == attack),
                            liphones == iphone,
                        )
                    ]
                    plt.scatter(
                        feat[:, 0],
                        feat[:, 1],
                        color=c,
                        marker="o",
                    )
                    n += 1
                    total_patches.append(
                        mpatches.Patch(
                            color=c, label=f"iPhone: {iphone} Attack: {attack}"
                        )
                    )

    plt.legend(handles=total_patches)
    plt.tight_layout()
    os.makedirs(
        "<Root-Directory>",
        exist_ok=True,
    )
    plt.savefig(
        os.path.join(
            "<Root-Directory>",
            "pca_" + tiphone + "_" + fname + ".png",
        )
    )


for iphone in iphones:
    plot_features(FEAT1_EMB, LABEL, IPHONE, ATTACK, "feature1", iphone)
    plot_features(FEAT2_EMB, LABEL, IPHONE, ATTACK, "feature2", iphone)
