"""[guide P3.1] LightGBM trade-count reconciliation. Committed artifact: lgbm_selection_reconciliation.json"""
import json; from pathlib import Path; import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/"TP2MVTransformer/data"; OUT=ROOT/"TP2MVTransformer/tables/action_list"
lg=pd.read_parquet(D/"lgbm_final_test_predictions.parquet"); lg["signal_date"]=pd.to_datetime(lg["signal_date"])
if "is_valid" in lg: lg=lg[lg["is_valid"].astype(bool)]
lg=lg[lg["strategy"].isin(["T1","K2"])]; cross=lg[lg["predicted_prob"]>=0.5]
per=cross.groupby("episode_id")["strategy"].nunique(); both=int((per==2).sum()); one=int((per==1).sum())
one_action=int(cross.sort_values(["predicted_prob","strategy"],ascending=[False,True]).groupby("episode_id").first().shape[0])
rec={"total_leg_selections_at_0p5":int(len(cross)),"one_action_per_episode":one_action,"episodes_both_legs_cross":both,
 "episodes_one_leg_cross":one,"selected_trades_file_convention":19636,"diff_dual_selection":int(len(cross)-one_action),
 "explanation":"per-strategy classifiers; selected-trades file keeps both legs (dual-selection); one-action keeps higher-prob leg."}
json.dump(rec,open(OUT/"lgbm_selection_reconciliation.json","w"),indent=2,default=float)
print(f"leg-selections={len(cross):,} one-action={one_action:,} both-cross={both:,}")
