import os
import glob
import open3d as o3d
import numpy as np

from torch.utils.data import Dataset


class Custom(Dataset):
    def __init__(
        self,
        iphone,
        attack,
        partition="Train",
        rdir="/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/Intra/",
        num_points=8192,
    ):
        self.basedir = os.path.join(rdir, iphone, attack)
        self.categories = ["mask", "bona"]
        self.filepaths = []
        self.category_idxs = []
        self.num_points = num_points

        for idx, category in enumerate(self.categories):
            category_path = os.path.join(
                self.basedir, partition, category, "**", "*.ply"
            )
            print("Fetching:", category_path)
            paths = glob.glob(category_path, recursive=True)
            print("Fetched:", len(paths))
            self.filepaths += paths
            self.category_idxs += [idx] * len(paths)

        if not len(self.filepaths):
            print("Loading data for:", partition, len(self.filepaths), iphone, attack)
            exit()

    def __len__(self):
        return len(self.category_idxs)

    def load_point_cloud_from_mesh(self, infile):
        mesh = o3d.io.read_triangle_mesh(infile, print_progress=False)
        pc = mesh.sample_points_uniformly(self.num_points)
        return pc.points

    def normalize_points(self, point_cloud):
        points = np.array(point_cloud).astype("float32")
        points = (points - np.mean(points, axis=0)) / (np.std(points, axis=0) + 1e-6)
        return points

    def __getitem__(self, index):
        cat_id = self.category_idxs[index]
        labels = [0] * len(self.categories)
        labels[cat_id] = 1
        category = np.array([cat_id])

        try:
            #             print("trying:", self.filepaths[index])
            data = self.load_point_cloud_from_mesh(self.filepaths[index])
            #             print("loaded:", self.filepaths[index])
            data = self.normalize_points(data)
        except Exception as e:
            print(e)
            return None
        return data, category

    @staticmethod
    def number_classes():
        return 2


# data_set = Custom("iPhone11", "Attack_1", partition="Train")
# for data, label in data_set:
#     print(type(data), type(label))
#     print(data.shape, label.shape)
#     print(data, label)
#     break
