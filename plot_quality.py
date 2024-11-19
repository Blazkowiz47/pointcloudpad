import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np

attacks = [f"Attack_{i}" for i in range(1, 9)]
cids = ["bona", "mask"]
iphones = ["iPhone12", "iPhone11"]


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

metrics = ["planarity", "linearity"]

for iphone in tqdm(iphones):
    colour = ["red", "blue", "green", "yellow", "purple", "orange"]
    attk_data = [[] for _ in metrics]
    label_data = []
    plt.figure()
    for i in range(8):
        for j, metric in enumerate(metrics):
            attk = (
                "quality_results/"
                + "_".join([cids[1], iphone, attacks[i], metric])
                + ".npy"
            )
            attk_data[j].extend(np.load(attk).tolist())
            new_data = len(np.load(attk).tolist())

        label_data.extend([i] * new_data)

    for i in [-1]:
        for j, metric in enumerate(metrics):
            attk = (
                "quality_results/"
                + "_".join([cids[0], iphone, attacks[i], metric])
                + ".npy"
            )
            attk_data[j].extend(np.load(attk).tolist())
            new_data = len(np.load(attk).tolist())

        label_data.extend([8] * new_data)

    attk_data = np.array(attk_data).transpose()

    # X_embedded = TSNE(n_components=2, learning_rate='auto',
    #                 init='random', perplexity=3).fit_transform(attk_data)
    X_embedded = PCA(n_components=2).fit_transform(attk_data)
    print(X_embedded.shape)
    label_data = np.array(label_data)
    for i in range(9):
        temp = label_data == i
        plt.scatter(
            X_embedded[temp, 0],
            X_embedded[temp, 1],
            label=attacks[i] if i != 8 else "bonafide",
            alpha=0.5 if i not in [6, 7] else 1,
            #             marker=".",
        )

    plt.legend(loc="upper right")
    plt.title(f"PCA on {iphone}")
    plt.xlim(-0.2, 0.3)
    plt.tight_layout()
    plt.savefig(
        f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/pca_{iphone}.png",
        bbox_inches="tight",
    )
    # plt.savefig(f'plot_test_tsne_{iphone}.png')

# for metric in ['linearity', 'planarity']:
#     colour = [ "red", "blue", "green", "yellow", "purple", "orange"]
#     plt.figure()
#     for i in range(8):
#         attk  ='quality_results/'+'_'.join([cids[1], iphones[0], attacks[i], metric]) + '.npy'
#         attk = np.load(attk).tolist()
#         plt.hist(attk, label=attacks[i], histtype='step')

#     plt.xlabel(metric.capitalize())
#     plt.legend(attacks)
#     plt.tight_layout()
#     plt.savefig(f'plot_test_{metric}.png')
