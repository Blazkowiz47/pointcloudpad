import glob
import os
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.quality_estimation import get_features

iphones = ["iPhone11", "iPhone12"]


def get_quality_estimates_for_dir(args: Tuple[int, str]) -> Dict[str, Any]:
    process_num, rdir = args
    files = glob.glob(os.path.join(rdir, "**", "*.ply"), recursive=True)
    results: List[Dict[str, Any]] = []
    for file in tqdm(files, position=process_num):
        measures = get_features(file)
        results.append(measures)

    final_results: Dict[str, List[Any]] = {}

    for result in results:
        for k, v in result.items():
            if k not in final_results:
                final_results[k] = []
            if k == "time":
                final_results[k].append(v)
                continue

            final_results[k].append(v["mean"].tolist())

    return final_results


total_means = {
    "iphone": [],
    "attack": [],
    "cid": [],
}


def driver() -> None:
    args = []

    ROOT_DIR = "/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/"
    attacks = ["Display-Attack", "Print-Attack"]
    for iphone in iphones:
        for attack in attacks:
            for cid in ["mask", "bona"]:
                rdir = os.path.join(ROOT_DIR, iphone, attack, "test", cid)
                args.append((rdir, iphone, attack, cid))

    ROOT_DIR = "/cluster/nbl-users/Shreyas-Sushrut-Raghu/Intra/"
    attacks = [f"Attack_{i}" for i in range(1, 7)]
    for iphone in iphones:
        for attack in attacks:
            for cid in ["mask"]:
                rdir = os.path.join(ROOT_DIR, iphone, attack, "test", cid)
                args.append((rdir, iphone, attack, cid))

    with Pool(20) as p:
        os.makedirs("quality_results", exist_ok=True)
        results = p.map(
            get_quality_estimates_for_dir, enumerate([arg[0] for arg in args])
        )
        for result, arg in zip(results, args):
            rdir, iphone, attack, cid = arg
            total_means["iphone"].append(iphone)
            total_means["attack"].append(attack)
            total_means["cid"].append(cid)
            for k, v in result.items():
                if k not in total_means:
                    total_means[k] = []
                np.save(f"quality_results/{cid}_{iphone}_{attack}_{k}.npy", np.array(v))

                total_means[k].append((
                    round(np.mean(v).tolist(), 2),
                    round(np.std(v).tolist(), 2),
                ))


if __name__ == "__main__":
    driver()
    df = pd.DataFrame(total_means)
    print(df)
    df.to_csv("./total_variation.csv")
