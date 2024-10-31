import time
from typing import Any, Dict

from pyntcloud import PyntCloud
from skimage import color

from .nss_functions import get_color_nss_param, get_geometry_nss_param


def get_feature_vector(objpath):
    cloud = PyntCloud.from_file(objpath)
    k_neighbors = cloud.get_neighbors(k=10)
    ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    cloud.add_scalar_field("curvature", ev=ev)
    cloud.add_scalar_field("anisotropy", ev=ev)
    cloud.add_scalar_field("linearity", ev=ev)
    cloud.add_scalar_field("planarity", ev=ev)
    cloud.add_scalar_field("sphericity", ev=ev)
    curvature = cloud.points["curvature(11)"].to_numpy()
    anisotropy = cloud.points["anisotropy(11)"].to_numpy()
    linearity = cloud.points["linearity(11)"].to_numpy()
    planarity = cloud.points["planarity(11)"].to_numpy()
    sphericity = cloud.points["sphericity(11)"].to_numpy()
    nss_params = []
    # compute color nss features
    #     for tmp in [l, a, b]:
    #         params = get_color_nss_param(tmp)
    #         # flatten the feature vector
    #         nss_params = nss_params + [i for item in params for i in item]

    # compute geomerty nss features
    for tmp in [curvature, anisotropy, linearity, planarity, sphericity]:
        params = get_geometry_nss_param(tmp)
        # flatten the feature vector
        nss_params = nss_params + [i for item in params for i in item]
    return nss_params


def get_features(fname: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    start = time.time()
    features = get_feature_vector(fname)
    end = time.time()
    time_cost = end - start

    cnt = 0
    for feature_domain in [
        "curvature",
        "anisotropy",
        "linearity",
        "planarity",
        "sphericity",
    ]:
        results[feature_domain] = {}
        for param in [
            "mean",
            "std",
            "entropy",
        ]:
            results[feature_domain][param] = round(features[cnt], 3)
            cnt = cnt + 1

    results["time"] = round(time_cost, 3)
    return results
