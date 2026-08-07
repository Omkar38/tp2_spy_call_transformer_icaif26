"""Comprehensive outputs PDF: every rerun's results + what was done, so it's understandable.
Loads the committed artifacts under TP2MVTransformer/tables/action_list/ and emits one readable report.
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ACT = HERE.parent/"TP2MVTransformer"/"tables"/"action_list"
def L(f):
    try: return json.load(open(ACT/f))
    except Exception: return {}
d13=L("lgbm_f48_comparison.json"); d16=L("exdiv_exclusion_rerun.json"); d21=L("calibration_by_year.json")
d22=L("clustered_inference.json"); d23=L("random_selection_distribution.json"); d25=L("spy_decomposition.json")
d26=L("denominator_diagnostics.json"); d27=L("quote_filter_ladder.json"); d20=L("value_rule_comparison.json")
d18=L("integer_sizing.json"); dnm=L("newmodel_reruns.json"); dcb=L("newmodel_cash_budgeted.json"); d14=L("forward_robustness.json")
def m(x): return f"\\${x/1e6:.2f}M" if abs(x)>=1e6 else f"\\${x/1e3:.0f}K"

# ex-div rows (NEW models)
ex = dnm.get("exdiv_newmodel", {})
exrows = "\n".join(f"\\code{{{k}}} & {v['full']['sharpe']:.2f} & {v['excl_exdiv']['sharpe']:.2f} & {v['pct_exposed']*100:.0f}\\% \\\\" for k,v in ex.items())
# new-model cash-budgeted rows
cb = dcb.get("models", {})
cbrows = "\n".join(f"\\code{{{k}}} & {v['premium_cagr']*100:.1f}\\% & {v['integer_cagr']*100:.1f}\\% & {v['margin_cagr']*100:.1f}\\% \\\\" for k,v in cb.items())
# clustered rows
cl = d22.get("models", {})
clrows = "\n".join(f"\\code{{{k}}} & {m(v['pnl_point'])} & [{m(v['pnl_ci_dateblock'][0])}, {m(v['pnl_ci_dateblock'][1])}] & {'incl.\\ 0' if v['pnl_ci_dateblock'][0]<=0<=v['pnl_ci_dateblock'][1] else 'excl.\\ 0'} \\\\" for k,v in cl.items())
# newmodel table2
nm = dnm.get("table2_newmodel", {})
nmrows = "\n".join(f"\\code{{{k}}} & {v['n']:,} & {m(v['pnl'])} & {v['sharpe']:.2f} & {v['pos_years']}/6 \\\\" for k,v in nm.items())
# decomposition rungs
dl = {r.get("rung",""):r for r in d25.get("unconditional_ladder",[]) if isinstance(r,dict)}
def ladval(key,f):
    r=dl.get(key,{}); return f(r) if r else "--"

TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs,array,xcolor,amsmath,longtable}
\usepackage{hyperref}\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\newcommand{\code}[1]{\texttt{\small #1}}
\setlength{\parskip}{4pt}
\title{\vspace{-1.2em}TP2 ICAIF --- Rerun Outputs, Explained\\
\large Every professor-task rerun: what it does, its result, and where it lives}
\author{SPY American Call TP2 Project}\date{August 7, 2026}
\begin{document}\maketitle\vspace{-1.0em}

\noindent This report collects the outputs of every rerun executed for the professor's revision queue.
Each reproduced its committed gate first (a failed gate stops the task), copied \code{metrics()} /
\code{sel\_by\_thresh} verbatim from \code{60\_action\_list\_compute.py}, and is committed as
\code{JSON + CSV + script}. Per your directive, model-dependent analyses use the \textbf{new perfect-pipeline
models}, except \textbf{LightGBM} which stays the \textbf{old 149-feature} model (no leak).

\section{New-model results (the footing)}
Active-day Sharpe, one action per episode, 2021--2026:
\begin{center}\footnotesize\begin{tabular}{l r r r c}
\toprule Model & N & P\&L & Sharpe & +yrs \\\midrule
""" + nmrows + r"""
\bottomrule\end{tabular}\end{center}

\section{brief-25 --- SPY decomposition ladder (the headline finding)}
How the SPX$\to$SPY reversal accrues, unconditional (gate: net = $-$\$12.52M, reproduced):
\begin{center}\footnotesize\begin{tabular}{l r r r}
\toprule Rung & N & P\&L & win \\\midrule
SPY bid/ask (pre-commission) & """ + ladval("1. SPY bid/ask (pre-commission)", lambda r:f"{r['n']:,}") + r""" & """ + ladval("1. SPY bid/ask (pre-commission)", lambda r:m(r['pnl'])) + r""" & """ + ladval("1. SPY bid/ask (pre-commission)", lambda r:f"{r['win_rate']*100:.1f}\\%") + r""" \\
SPY +commissions (net) & """ + ladval("2. SPY +commissions (unconditional net)", lambda r:f"{r['n']:,}") + r""" & """ + ladval("2. SPY +commissions (unconditional net)", lambda r:m(r['pnl'])) + r""" & """ + ladval("2. SPY +commissions (unconditional net)", lambda r:f"{r['win_rate']*100:.1f}\\%") + r""" \\
\bottomrule\end{tabular}\end{center}
\noindent\textbf{Finding:} at conservative bid/ask the SPY structure still wins """ + ladval("1. SPY bid/ask (pre-commission)", lambda r:f"{r['win_rate']*100:.1f}\\%") + r""" (+\$61M,
near SPX's 95--100\%); the \$73.5M commission drag drives the $-$\$12.52M loss. The reversal is essentially a
\emph{single friction} (commissions), which the paper deliberately avoids attributing --- \textbf{flag for PI}.

\section{brief-23 --- 1000-draw random (day- vs trade-selection)}
Three matching levels (L1 reproduces the paper's $-$\$0.77M gate):
\begin{center}\footnotesize\begin{tabular}{l r r}
\toprule Level & mean P\&L & P(random $\ge$ LightGBM / plain) \\\midrule
L1 count-only & """ + (m(d23['levels']['L1_count_only']['pnl_mean']) if d23 else "--") + r""" & """ + (f"{d23['levels']['L1_count_only']['p_random_ge_model']['LightGBM']:.2f} / {d23['levels']['L1_count_only']['p_random_ge_model']['Transformer-plain-5']:.2f}" if d23 else "--") + r""" \\
L2 +year/denom & """ + (m(d23['levels']['L2_year_denom']['pnl_mean']) if d23 else "--") + r""" & """ + (f"{d23['levels']['L2_year_denom']['p_random_ge_model']['LightGBM']:.2f} / {d23['levels']['L2_year_denom']['p_random_ge_model']['Transformer-plain-5']:.2f}" if d23 else "--") + r""" \\
L3 +per-day (full) & """ + (m(d23['levels']['L3_perday_full']['pnl_mean']) if d23 else "--") + r""" & """ + (f"{d23['levels']['L3_perday_full']['p_random_ge_model']['LightGBM']:.2f} / {d23['levels']['L3_perday_full']['p_random_ge_model']['Transformer-plain-5']:.2f}" if d23 else "--") + r""" \\
\bottomrule\end{tabular}\end{center}
\noindent\textbf{Finding:} at L3 (matched active-day structure) random beats LightGBM and plain transformer
most of the time --- their edge is \emph{day-selection}, not within-day trade-selection. \textbf{Flag for PI.}

\section{brief-16 --- ex-dividend-short exclusion}
Excluding trades whose short leg spans an ex-date (16--27\% exposed):
\begin{center}\footnotesize\begin{tabular}{l r r r}
\toprule Model & full Sharpe & excl.\ Sharpe & \% exposed \\\midrule
""" + exrows + r"""
\bottomrule\end{tabular}\end{center}
Excluding improves most models; all stay positive.

\section{brief-22 --- date/episode-clustered inference}
Date-block-bootstrap 95\% P\&L intervals (point estimates reproduce Table 2):
\begin{center}\footnotesize\begin{tabular}{l r l l}
\toprule Model & P\&L & 95\% CI & significance \\\midrule
""" + clrows + r"""
\bottomrule\end{tabular}\end{center}
LightGBM/MLP-5 CIs include zero (concentration-driven); transformers/logistic exclude zero.
ALiBi$-$plain difference CI includes zero --- no significant architecture difference.

\section{Other reruns (one line each)}
\begin{itemize}\itemsep2pt
\item \textbf{brief-13} LightGBM-on-48 = 1.71 (6/6) vs on-149 = 1.25 --- \textbf{+0.46 home-turf boost} confirms the 48 is circular; report LightGBM on 149.
\item \textbf{brief-21} per-year calibration; Brier/ECE match Table 4; MLP selected mean-$p$ 0.709 vs win 33\% (temporal calibration failure).
\item \textbf{brief-26} denominators; 2013 = 495 viol/1k vs 16.8 median + 100\% weekly-expiry $\Rightarrow$ coverage anomaly (justifies exclusion).
\item \textbf{brief-27} quote-quality ladder; individual filters survive, combined strict keeps $\sim$5--7\% and collapses the edge.
\item \textbf{brief-20} value rule $E[R|X]{>}0$ = """ + (m(d20['value_rule']['pnl'])+f", Sharpe {d20['value_rule']['sharpe']:.2f}" if d20 else "--") + r""" vs p$\ge$0.5 = \$682K/1.25 --- more gross P\&L, lower risk-adjusted.
\item \textbf{brief-14} forward robustness: """ + (f"{d14['retention_pct']*100:.0f}\\%" if d14 else "--") + r""" of episodes keep their adjusted-strike targets under an independent forward, and the P\&L \textbf{sign is unchanged} for every model.
\item \textbf{Task 0.2} two-rule + fiveseed re-anchor: artifacts complete, gate 24,977/\$3.43M/1.66.
\item \textbf{Task 0.3} equity-curves figure: panel (a) = Table 2, panel (b) = Table 3.
\end{itemize}

\section{brief-18 + brief-19 on the NEW models (cash-budgeted)}
Cash-budgeted CAGR on the new-model trade sets (LightGBM = old 149, anchors Table 3):
\begin{center}\footnotesize\begin{tabular}{l r r r}
\toprule Model & premium CAGR & integer CAGR & margin CAGR (Reg-T) \\\midrule
""" + cbrows + r"""
\bottomrule\end{tabular}\end{center}
\noindent New-pipeline transformers are far more capital-efficient than the old committed ones (Table 3's 0.6\%).
Integer-contract sizing and conservative Reg-T margin are sensitivities (margin is an upper-bound / lower-CAGR
model; OCC netting would be less severe).

\section{Only one item remains}
\begin{itemize}\itemsep2pt
\item \textbf{New-model calibration} (full-population Brier/ECE) --- needs the full predicted probabilities the
new pipeline did not save (it stored only selected trades). The selected-tail (mean calibrated $p$, win rate)
\emph{is} provided in \code{newmodel\_reruns.json}. A one-line pipeline tweak to also dump full predictions closes it.
\end{itemize}

\section{Where everything lives (traceability)}
All artifacts are committed under \code{TP2MVTransformer/tables/action\_list/} as \code{<task>.json} +
\code{<task>.csv}, generated by scripts \code{69--79}, each commit stating its gate result. New-model trades:
\code{FeatureDataDocs\_2026-08-04/nested/trades\_\{lgbm,logistic,mlp5,transformer\_alibi,transformer\_plain\}}.
\textbf{No v5 tex has been edited} --- tex insertions are held for one PI-reviewed pass, because the
commission and day-selection findings could reshape the paper's framing.
\end{document}
"""
(HERE/"rerun_outputs_summary.tex").write_text(TEX)
print("wrote rerun_outputs_summary.tex")
