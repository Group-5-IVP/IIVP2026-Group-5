import json, time
from torchvision.transforms import v2
import torch
import torch.nn.functional as F
import pandas as pd
from Dataset import train_csv_path
from augmentation import random_stroke
from datasetStats import mean, std
from models import SimpleCNN
from validation import simple_validation
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "aug_search.jsonl"

BASE = [v2.Grayscale(1), v2.Resize((32, 32))]
TO_TENSOR = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
NORM = v2.Normalize(mean=[mean], std=[std])

DEFAULTS = {
    "affine":  {"degrees": 10, "translate": (0.1, 0.1), "scale": (0.9, 1.1), "shear": 5},
    "elastic": {"alpha": 8.0, "sigma": 4.0},
    "jitter":  {"brightness": 0.2, "contrast": 0.2},
    "stroke":  {"prob": 0.5, "k": 3},
    "erase":   {"p": 0.25, "scale": (0.02, 0.15)},
}

def cfg(name, **overrides):
    return {"name": name, **{**DEFAULTS, **overrides}}

def cfg_off(name, *off_keys):
    return {"name": name, **{k: (None if k in off_keys else v) for k, v in DEFAULTS.items()}}

CONFIGS = [
    cfg("baseline"),

    # leave-one-out (which augment matters?)
    cfg_off("no_affine",  "affine"),
    cfg_off("no_elastic", "elastic"),
    cfg_off("no_jitter",  "jitter"),
    cfg_off("no_stroke",  "stroke"),
    cfg_off("no_erase",   "erase"),
    cfg_off("nothing", "affine", "elastic", "jitter", "stroke", "erase"),

    # magnitude sweeps for affine
    cfg("affine_weak",   affine={"degrees": 5,  "translate": (0.05, 0.05), "scale": (0.95, 1.05), "shear": 2}),
    cfg("affine_strong", affine={"degrees": 20, "translate": (0.15, 0.15), "scale": (0.8, 1.2),  "shear": 10}),

    # elastic sweeps
    cfg("elastic_weak",   elastic={"alpha": 4.0, "sigma": 4.0}),
    cfg("elastic_strong", elastic={"alpha": 12.0, "sigma": 4.0}),

    # stroke sweeps
    cfg("stroke_always", stroke={"prob": 1.0, "k": 3}),
    cfg("stroke_k5",     stroke={"prob": 0.5, "k": 5}),

    # erasing sweeps
    cfg("erase_more",  erase={"p": 0.5, "scale": (0.02, 0.15)}),
    cfg("erase_big",   erase={"p": 0.25, "scale": (0.05, 0.3)}),

    # jitter sweeps
    cfg("jitter_strong", jitter={"brightness": 0.4, "contrast": 0.4}),
]

def make_transform(affine=None, elastic=None, jitter=None, stroke=None, erase=None):
    pil = []
    if affine:  pil.append(v2.RandomAffine(**affine))
    if elastic: pil.append(v2.ElasticTransform(**elastic))
    if jitter:  pil.append(v2.ColorJitter(**jitter))

    tail = []
    if stroke: tail.append(v2.Lambda(lambda x, s=stroke: random_stroke(x, **s)))
    tail.append(NORM)
    if erase:  tail.append(v2.RandomErasing(**erase))

    return v2.Compose(BASE + pil + TO_TENSOR + tail)

def run_search(df, model_fn, configs, out=RESULTS_PATH, epochs=10):
    import models
    for cfg in configs:
        name = cfg["name"]
        cfg_params = {k: v for k, v in cfg.items() if k != "name"}
        models._build_train_transform = lambda c=cfg_params: make_transform(**c)
        t0 = time.time()
        acc = simple_validation(df, model_fn, epochs=epochs)
        with open(out, "a") as f:
            f.write(json.dumps({"name": name, "cfg": cfg_params, "acc": acc, "sec": int(time.time()-t0)}) + "\n")
        print(f"{name}: {acc:.2f}%")

if __name__ == '__main__':
    run_search(pd.read_csv(train_csv_path), SimpleCNN, CONFIGS, epochs=10)
