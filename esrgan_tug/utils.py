"""Config loading, seeding and image I/O shared by all scripts."""
import copy
import os
import random

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        cfg = deep_merge(base, cfg)
    for o in overrides or ():
        k, v = o.split("=", 1)
        set_by_path(cfg, k, yaml.safe_load(v))
    return cfg


def model_recipe(cfg, name):
    r = cfg["models"][name]
    if "inherit" in r:
        parent = model_recipe(cfg, r["inherit"])
        r = deep_merge(parent, {k: v for k, v in r.items() if k != "inherit"})
    return copy.deepcopy(r)


def scaled(n, cfg):
    return max(1, int(round(n * float(cfg.get("iters_scale", 1.0)))))


def run_dir(cfg, seed, *parts):
    return os.path.join(cfg["paths"]["runs"], f"seed{seed}", *parts)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(img[..., ::-1])


def imwrite_rgb(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.ascontiguousarray(img[..., ::-1]))


def list_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(exts)) if os.path.isdir(folder) else []
