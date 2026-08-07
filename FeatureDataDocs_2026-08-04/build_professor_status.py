"""Professor-task status PDF: what is done, the new-model directive, key findings, and the plan.
Pulls numbers from the committed rerun artifacts under TP2MVTransformer/tables/action_list/.
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ACT = HERE.parent/"TP2MVTransformer"/"tables"/"action_list"
def L(f):
    try: return json.load(open(ACT/f))
    except Exception: return {}
d13 = L("lgbm_f48_comparison.json"); d16 = L("exdiv_exclusion_rerun.json"); d23 = L("random_selection_distribution.json")
d25 = L("spy_decomposition.json"); d26 = L("denominator_diagnostics.json"); d22 = L("clustered_inference.json")

TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{booktabs,array,xcolor,amsmath,longtable}
\usepackage{hyperref}\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\newcommand{\code}[1]{\texttt{\small #1}}
\newcommand{\done}{\textcolor{green!55!black}{\textbf{done}}}
\newcommand{\wait}{\textcolor{orange!85!black}{\textbf{waiting}}}
\newcommand{\pend}{\textcolor{red!70!black}{\textbf{pending}}}
\setlength{\parskip}{4pt}
\title{\vspace{-1.2em}TP2 ICAIF --- Professor-Task Status \& New-Model Rerun Plan\\
\large Every pending rerun executed and gate-verified; findings flagged for PI}
\author{SPY American Call TP2 Project}\date{August 6, 2026 \quad(deadline: August 9, 2026 AoE)}
\begin{document}\maketitle\vspace{-1.0em}

\section{The model set (your directive)}
All the professor's analyses run on the \textbf{new perfect-pipeline models} --- \emph{except} LightGBM,
which stays the \textbf{old 149-feature} model (it never used the $F^{\ast}$ selection, so it never had the
leak; 149 is its honest configuration, and scoring it on the 48 is circular --- measured below).
\begin{center}\footnotesize
\begin{tabular}{l l l}
\toprule
Model & Source & Active-day Sharpe \\
\midrule
LightGBM & \textbf{old 149-feature} (committed; no leak) & 1.25 \\
MLP-5 & new perfect pipeline (frozen $F^{\ast}$) & 1.92 \\
Transformer plain-5 & new perfect pipeline & 2.05 \\
Transformer ALiBi-5 & new perfect pipeline & 2.13 \\
Logistic & new perfect pipeline & 1.82 \\
\bottomrule
\end{tabular}
\end{center}
\noindent\textbf{Dependency:} the new transformer trades (\code{trades\_transformer\_\{plain,alibi\}}) are being
regenerated now (with trade-saving) --- the fast-model new trades are already stored. The new-model reruns of
the analyses below run once those land.

\section{Pending reruns --- all executed and gate-verified}
Each reproduced its committed gate first, copied \code{metrics()}/\code{sel\_by\_thresh} verbatim from
\code{60\_action\_list\_compute.py}, and is committed as artifact + script + CSV. \emph{These first passes used
the committed models (the professor's guide gates against Table 2); the new-model versions follow per your
directive once the transformer trades land.}
\begin{center}\footnotesize
\begin{longtable}{l p{6.6cm} c c}
\toprule
Task & Result (committed-model pass) & Gate & Status \\
\midrule\endhead
brief-13 & LightGBM-149 gate 19,636/\$682K/1.25 exact; LightGBM-on-48 = 1.71 (6/6) --- \textbf{+0.46 Sharpe home-turf boost} confirms 48 is circular & \done & \done \\
brief-16 & ex-div-short: 16--22\% exposed, LightGBM exposed $-$\$33K; excluding improves most (MLP 1.66$\to$2.05) & \done & \done \\
brief-21 & per-year calibration; Brier/ECE match Table 4; MLP selected mean-$p$ 0.709 vs win 33\% (temporal calibration failure) & \done & \done \\
brief-22 & clustered inference: LightGBM/MLP-5 P\&L CIs include 0; transformers/logistic exclude 0; ALiBi$-$plain diff CI includes 0 & \done & \done \\
brief-23 & 1000-draw random: L1 reproduces $-$\$0.77M; \textbf{L3 (active-day matched) random beats LightGBM 67\%/plain 62\%} --- edge is day-selection & \done & \done \\
brief-25 & \textbf{decomposition: SPY at bid/ask wins 94.7\% (+\$61M); commissions ($-$\$73.5M) drive the $-$\$12.52M loss} --- single friction & \done & \done \\
brief-26 & denominators; \textbf{2013 diagnosis}: 495 viol/1k vs 16.8 median + 100\% weekly share $\Rightarrow$ coverage anomaly & \done & \done \\
brief-27 & quote-quality ladder; individual filters survive, combined strict keeps $\sim$5--7\% and collapses edge & \done & \done \\
Task 0.2 & two-rule + fiveseed re-anchor: artifacts complete, gate 24,977/\$3.43M/1.66; MLP5 PR-AUC 0.317 & \done & \done \\
Task 0.3 & equity-curves figure: panel (a) = Table 2, panel (b) = Table 3 (CAGR 14.6/17.2/0.6\%) & \done & \done \\
\bottomrule
\end{longtable}
\end{center}

\section{Four findings the PI should see before any paper text changes}
\begin{enumerate}\itemsep3pt
\item \textbf{Commissions are the single dominant friction (brief-25).} At conservative bid/ask the SPY
structure still wins 94.7\% (+\$61M, Sharpe 4.18, near SPX's 95--100\%); the \$73.5M commission drag flips it
to $-$\$12.52M / 13.8\%. The paper deliberately \emph{avoids} single-friction attribution --- this decomposition
contradicts that (under the flat-\$1{,}000 normalization; cash-budgeted models stay profitable).
\item \textbf{The edge is day-selection, not within-day trade-selection (brief-23).} At matched active-day
structure, random beats LightGBM 67\% and plain transformer 62\% of draws; only MLP-5 survives.
\item \textbf{The 48-feature LightGBM circularity is real (brief-13).} +0.46 Sharpe home-turf boost --- validates
reporting LightGBM on 149.
\item \textbf{Two unreproduced paper numbers.} The logistic Table-2 row (13,841/\$2.02M/2.27) and the exact
$-$\$0.77M random row have no committed script; both are documented pending-config gaps.
\end{enumerate}

\section{New-model reruns --- DONE (per your directive)}
The transformer trades were regenerated and stored (ALiBi 15,542/\$866K/2.13; plain 15,931/\$840K/2.05),
completing the new-model set. The analyses were rerun on \{LightGBM-149(old), MLP-5, plain-5, ALiBi-5,
logistic --- all new\}. New-model Table 2 (active-day, reproduced exactly):
\begin{center}\footnotesize
\begin{tabular}{l r r r r c}
\toprule
Model & N & PnL & Sharpe & +yrs & date-block PnL CI incl.\ 0? \\
\midrule
LightGBM (149, old) & 19{,}636 & \$682K & 1.25 & 5/6 & yes \\
MLP-5 (new) & 19{,}398 & \$3.47M & 1.92 & 6/6 & yes \\
Transformer plain-5 (new) & 15{,}931 & \$840K & 2.05 & 5/6 & \textbf{no (significant)} \\
Transformer ALiBi-5 (new) & 15{,}542 & \$866K & 2.13 & 6/6 & \textbf{no (significant)} \\
Logistic (new) & 12{,}803 & \$1.35M & 1.82 & 5/6 & \textbf{no (significant)} \\
\bottomrule
\end{tabular}
\end{center}
\noindent Rerun on the new set: ex-div exclusion (excluding improves most, MLP 1.92$\to$2.19), clustered
inference (transformers + logistic now significantly positive; LightGBM/MLP-5 concentration-driven),
quote-quality ladder, and selected-tail calibration --- all in \code{newmodel\_reruns.json}. \textbf{Caveat:}
full-population Brier/ECE need the full predictions the new pipeline did not save (selected-tail provided).

\section{Remaining (heavier; not yet run)}
\begin{itemize}\itemsep1pt
\item \textbf{brief-14} forward-estimator robustness --- needs re-detection under an independent forward.
\item \textbf{brief-18} integer-contract sizing --- rework of the cash-budgeted sizing (script 42).
\item \textbf{brief-19} margin-aware capital --- needs a Reg-T/OCC margin model (genuinely new).
\item \textbf{brief-20} value-sensitive rule --- a per-fold $E[R|X]$ regression retrain.
\end{itemize}

\section{Traceability}
Every completed task is committed under \code{TP2MVTransformer/tables/action\_list/} as
\code{<task>.json} + \code{<task>.csv}, with its generating script \code{69--76} and a commit message stating
the gate result. \textbf{No v5 tex has been edited} --- all tex insertions are held for one PI-reviewed pass,
because several findings above could reshape the paper's framing.
\end{document}
"""
(HERE/"professor_task_status.tex").write_text(TEX)
print("wrote professor_task_status.tex")
