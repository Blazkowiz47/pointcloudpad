import os
import numpy as np
from common_metrics import eer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import DetCurveDisplay


ROOT = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/PAD-Features/DifferentDual/Ablation_DualDGCNN_DifferentDual_3_5_c_k1_8"
train_iphone = ["iPhone11"]
train_attack = [f"Attack_{i}" for i in range(1, 7)]
test_iphone = ["iPhone12"]
test_attack = [f"Attack_{i}" for i in range(1, 7)]


def plot_custom(train_iphone, train_attack):
    plt.figure("DET_Curve")
    iphone = test_iphone[0]
    colors = cm.rainbow(np.linspace(0, 1, len(test_attack)))
    for attack in test_attack:
        real_scores = np.loadtxt(
            os.path.join(
                ROOT,
                f"trained_on_{train_iphone}_{train_attack}",
                f"gen_{iphone}_{attack}.txt",
            )
        )
        attack_scores = np.loadtxt(
            os.path.join(
                ROOT,
                f"trained_on_{train_iphone}_{train_attack}",
                f"atk_{iphone}_{attack}.txt",
            )
        )
        #         print(
        #             max(real_scores.max(), attack_scores.max()),
        #             min(real_scores.min(), attack_scores.min()),
        #         )
        #         print(
        #             real_scores.min(),
        #             real_scores.max(),
        #             attack_scores.min(),
        #             attack_scores.max(),
        #         )

        thresholds = np.linspace(0, 1, 10001)
        eer_value = eer(real_scores, attack_scores)
        apcers = []
        bpcers = []
        for threshold in thresholds:
            bpcer = (
                np.sum(np.where(real_scores <= threshold, 1, 0))
                * 100
                / real_scores.shape[0]
            )
            apcer = (
                np.sum(np.where(attack_scores > threshold, 1, 0))
                * 100
                / attack_scores.shape[0]
            )
            apcers.append(apcer)
            bpcers.append(bpcer)

        plt.plot(
            apcers,
            bpcers,
            label=f"{attack}: {eer_value:.2f}",
        )
    #         print(eer_value)
    #         plt.plot(np.log10(eer_value * 100), np.log10(eer_value * 100), marker="o")

    plt.title(f"Trained on {train_iphone} {train_attack}")
    plt.grid(linestyle="--")
    plt.xlabel("APCER (%)")
    plt.ylabel("BPCER (%)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xscale("log")
    plt.yscale("log")
    x = np.linspace(0, 100, 10000)
    plt.plot(x, x, linestyle="--", color="black")

    xs = [0.1, 1, 10, 100]
    labels = [str(x) for x in xs]
    plt.xticks(xs, labels)
    plt.yticks(xs, labels)
    plt.legend(loc="lower right")
    plt.axis("square")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/plots",
            f"DET_{train_iphone}_{train_attack}.png",
        ),
    )
    plt.cla()
    plt.clf()
    plt.close()


def plot(tiphone, tattack):
    _, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    for iphone in test_iphone:
        for attack in test_attack:
            real_scores = np.loadtxt(
                os.path.join(
                    ROOT,
                    f"trained_on_{tiphone}_{tattack}",
                    f"gen_{iphone}_{attack}.txt",
                )
            )
            attack_scores = np.loadtxt(
                os.path.join(
                    ROOT,
                    f"trained_on_{tiphone}_{tattack}",
                    f"atk_{iphone}_{attack}.txt",
                )
            )
            eer_value = eer(real_scores, attack_scores)
            pred = np.concatenate((real_scores, attack_scores), axis=0)
            label = np.concatenate(
                (
                    np.ones((real_scores.shape[0],)),
                    np.zeros((attack_scores.shape[0],)),
                ),
                axis=0,
            )

            DetCurveDisplay.from_predictions(label, pred, ax=ax, name=attack)

            if not eer_value:
                plt.plot(eer_value, eer_value, marker="o")

    ax.set_title(f"Trained on {tiphone} {tattack}")
    ax.grid(linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/plots",
            f"DET_{tiphone}_{tattack}.png",
        ),
    )

    plt.show()


for tiphone in train_iphone:
    for tattack in train_attack:
        plot_custom(tiphone, tattack)
