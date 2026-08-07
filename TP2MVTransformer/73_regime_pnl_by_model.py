"""[guide P2.4/brief-15] Per-model regime P&L. Gate: each model's R1/R2/R3 sums to its Table 2 total.
Committed artifact: regime_pnl_by_model.json"""
import json; from pathlib import Path; import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/"TP2MVTransformer/data"; OUT=ROOT/"TP2MVTransformer/tables/action_list"
reg=pd.read_parquet(ROOT/"data/ml/episode_regime_labels.parquet")[["episode_id","regime"]]
reg["R"]=reg["regime"].str.startswith("R1").map({True:"R1"}).fillna(reg["regime"].map({"R2_dividend_robust":"R2","R3_unresolved":"R3"}))
GATE={"MLP (5)":("mlp_5seed_ensemble_selected_trades.parquet","target_norm_pnl",3433936),
 "Transformer plain (5)":("mv_no_alibi_5seed_selected_trades.parquet","target_norm_pnl",695173),
 "LightGBM (1)":("lgbm_final_t1k2_selected_trades.parquet","norm_pnl",682411)}
out={}
for m,(f,pc,tot) in GATE.items():
    df=pd.read_parquet(D/f)
    if "selected" in df: df=df[df["selected"].astype(bool)]
    df=df.merge(reg[["episode_id","R"]],on="episode_id",how="left")
    by={r:float(g[pc].sum()) for r,g in df.groupby("R")}
    out[m]={"R1":by.get("R1",0),"R2":by.get("R2",0),"R3":by.get("R3",0),"sum":sum(by.values()),"table2_total":tot,"gate_ok":bool(abs(sum(by.values())-tot)<1000)}
    assert out[m]["gate_ok"], f"GATE FAIL {m}"
json.dump(out,open(OUT/"regime_pnl_by_model.json","w"),indent=2,default=float); print("all gates passed:",all(v["gate_ok"] for v in out.values()))
