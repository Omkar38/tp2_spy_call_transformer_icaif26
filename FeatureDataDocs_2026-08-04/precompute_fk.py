"""Step A of the perfect transformer pipeline: precompute the per-fold ensemble F*_k
in a TORCH-FREE process (LightGBM and torch both bundle libomp on macOS -> segfault if
co-loaded). Feature selection is model-independent, so this same F*_k feeds BOTH the
ALiBi and Plain transformers.

Ensemble = deterministic LightGBM x 5 seeds on TRAINING YEARS ONLY -> average importance
-> top 48. Identical to the MLP-5 perfect pipeline (seed-set A = [1,2,3,4,5]).
Artifact: nested/fk_ensemble_perfect.json  ->  {test_year: [48 features]}
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/Users/jyotirpatel/Downloads/FinalQuant copy 2"); TP2 = ROOT/"TP2MVTransformer"; DAT = TP2/"data"
OUTD = Path(__file__).resolve().parent/"nested"; OUTD.mkdir(exist_ok=True)
DISC_SEEDS = [1, 2, 3, 4, 5]

spec = importlib.util.spec_from_file_location("bfd_pre", TP2/"01_build_feature_dictionary.py")
disc = importlib.util.module_from_spec(spec); spec.loader.exec_module(disc)
CANON = disc.FOLDS
disc.LGBM_PARAMS.update({"deterministic": True, "force_row_wise": True, "n_jobs": -1})

data = pd.read_parquet(DAT/"tp2_t1k2_token_dataset.parquet"); data["signal_date"] = pd.to_datetime(data["signal_date"])
data["signal_year"] = data["signal_date"].dt.year
data = data[data.strategy.isin(["T1", "K2"])]
ALL149 = pd.read_csv(TP2/"tables"/"feature_dictionary.csv")["feature"].tolist()

def ensemble_Fk(fold):
    trly = fold["train"]
    disc.FOLDS = [{"fold": "e", "train": trly[:-1], "val": [trly[-1]], "test": fold["test"]}]
    scores = []
    for s in DISC_SEEDS:
        disc.LGBM_PARAMS["random_state"] = s
        _, summ = disc.run_feature_discovery(data, ALL149)
        scores.append(summ.set_index("feature")["mean_score"])
    disc.FOLDS = CANON
    avg = pd.concat(scores, axis=1).mean(axis=1)
    return avg.sort_values(ascending=False).head(48).index.tolist()

fk = {}
for fold in CANON:
    yr = fold["test"][0]
    fk[str(yr)] = ensemble_Fk(fold)
    print(f"  fold test={yr}: F*_k selected (train-years {fold['train'][:-1]}, sel-val {fold['train'][-1]}) "
          f"-> {len(fk[str(yr)])} features; head={fk[str(yr)][:5]}", flush=True)

json.dump({"disc_seeds": DISC_SEEDS, "fk_by_year": fk,
           "config": "deterministic ensemble LightGBM x5, training-years-only, top-48"},
          open(OUTD/"fk_ensemble_perfect.json", "w"), indent=2)
print("wrote", OUTD/"fk_ensemble_perfect.json", flush=True)
