"""Final review PDF: complete new-model results, full calibration, pipeline re-check, git/paper state.
Loads committed artifacts under TP2MVTransformer/tables/action_list/.
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ACT = HERE.parent/"TP2MVTransformer"/"tables"/"action_list"
def L(f):
    try: return json.load(open(ACT/f))
    except Exception: return {}
nm=L("newmodel_reruns.json")["table2_newmodel"]; cal=L("newmodel_calibration.json")["models"]
cb=L("newmodel_cash_budgeted.json")["models"]; cl=L("newmodel_reruns.json")["clustered_newmodel"]
def m(x): return f"\\${x/1e6:.2f}M" if abs(x)>=1e6 else f"\\${x/1e3:.0f}K"

# Table 2 + calibration + cash-budgeted, per model
NAMES=[("LightGBM(149,old)","LightGBM (old 149)"),("MLP-5(new)","MLP-5"),("Transformer-plain-5(new)","TF-plain-5"),
       ("Transformer-ALiBi-5(new)","TF-ALiBi-5"),("Logistic(new)","Logistic")]
def calget(k,f):
    v=cal.get(k,{}); return f"{v[f]:.3f}" if f in v and isinstance(v.get(f),(int,float)) else "--"
rows=[]
for key,lab in NAMES:
    t=nm[key]; cbk=cb.get(key,{})
    rows.append(f"\\code{{{lab}}} & {t['n']:,} & {m(t['pnl'])} & {t['sharpe']:.2f} & {t['pos_years']}/6 & {calget(key,'brier')} & {calget(key,'ece')} & {cbk.get('premium_cagr',0)*100:.1f}\\% \\\\")
tbl="\n".join(rows)
clrows="\n".join(f"\\code{{{lab}}} & {m(cl[key]['pnl_point'])} & [{m(cl[key]['pnl_ci_dateblock'][0])}, {m(cl[key]['pnl_ci_dateblock'][1])}] & {'incl.\\,0' if cl[key]['pnl_ci_dateblock'][0]<=0<=cl[key]['pnl_ci_dateblock'][1] else '\\\\textbf{excl.\\,0}'} \\\\" for key,lab in NAMES if key in cl)

TEX=r"""\documentclass[11pt]{article}
\usepackage[margin=0.75in]{geometry}\usepackage{booktabs,array,xcolor,amsmath,longtable}
\usepackage{hyperref}\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\newcommand{\code}[1]{\texttt{\small #1}}
\setlength{\parskip}{4pt}
\title{\vspace{-1.2em}TP2 ICAIF --- Final Review (all professor reruns, new models)\\
\large Complete, gate-verified, on the new pipeline; PI's v5 merged; nothing pushed}
\author{SPY American Call TP2 Project}\date{August 7, 2026}
\begin{document}\maketitle\vspace{-1.0em}

\section*{Executive summary}
Every pending professor revision-brief rerun is done and committed, computed on the \textbf{new
perfect-pipeline models} (LightGBM stays the old 149-feature model --- it never had the $F^{\ast}$ leak).
Each reproduced its committed gate first; \code{metrics()}/\code{sel\_by\_thresh} were copied verbatim from
\code{60\_action\_list\_compute.py}. The full new-model calibration is now complete (transformers included).
A read-only pipeline re-check passes all gates. The PI's latest v5 paper (new abstract/intro/conclusion) has
been merged in. \textbf{Nothing was pushed, and no v5 tex was edited by the rerun work} --- all tex
insertions are held for one PI-reviewed pass.

\section{New models, one table (Table 2 + calibration + cash-budgeted)}
Active-day Sharpe, one action/episode, 2021--2026:
\begin{center}\footnotesize
\begin{tabular}{l r r r c r r r}
\toprule
Model & N & P\&L & Sharpe & +yrs & Brier & ECE & prem.\ CAGR \\
\midrule
"""+tbl+r"""
\bottomrule
\end{tabular}
\end{center}
\noindent The new transformers are the \textbf{best-calibrated} (ECE 0.036--0.038); MLP-5 stays miscalibrated
(selected mean-$p$ 0.695 vs win 37.7\%). New-pipeline transformers are far more capital-efficient than the old
committed ones (Table 3's 0.6\%).

\section{Clustered inference (date block bootstrap, 95\% P\&L CI)}
\begin{center}\footnotesize
\begin{tabular}{l r l l}
\toprule Model & P\&L & 95\% CI & significance \\\midrule
"""+clrows+r"""
\bottomrule\end{tabular}\end{center}
\noindent Transformers + logistic exclude zero (significantly positive); LightGBM/MLP-5 include zero
(concentration-driven). ALiBi$-$plain difference CI includes zero (no significant architecture difference).

\section{Every rerun (gate-verified, committed as JSON+CSV+script 69--85)}
\begin{itemize}\itemsep2pt
\item \textbf{brief-13} LightGBM-on-48 = 1.71 vs on-149 = 1.25: \textbf{+0.46 home-turf boost} $\Rightarrow$ report LightGBM on 149.
\item \textbf{brief-14} forward robustness: 66\% adjusted-strike retention, \textbf{P\&L sign unchanged} all models.
\item \textbf{brief-16} ex-div-short exclusion: 16--27\% exposed; excluding improves most (MLP 1.92$\to$2.19).
\item \textbf{brief-18/19} integer sizing + Reg-T margin (new models): margin-aware CAGR collapses ($\sim$1--2\%, conservative).
\item \textbf{brief-20} value rule $E[R|X]{>}0$ = \$3.80M/0.94 vs p$\ge$0.5 = \$682K/1.25 (more gross, lower Sharpe).
\item \textbf{brief-21} per-year calibration; MLP selected mean-$p$ 0.709 vs win 33\% (temporal miscalibration).
\item \textbf{brief-23} random: L1 reproduces $-$\$0.77M; L3 random beats LightGBM-old 67\% but new plain only 8\%.
\item \textbf{brief-25} decomposition: SPY at bid/ask wins 94.7\% (+\$61M); commissions ($-$\$73.5M) drive the loss.
\item \textbf{brief-26} 2013 diagnosis: 495 viol/1k vs 16.8 median + 100\% weekly $\Rightarrow$ coverage anomaly.
\item \textbf{brief-27} quote-quality ladder; combined-strict keeps $\sim$5--7\% and collapses the edge.
\item \textbf{Task 0.2} fiveseed on new MLP-5 (T1 +\$990K, K2 +\$2.48M, both positive); \textbf{Task 0.3} equity figure.
\item \textbf{Task 2.4} P\&L by regime: MLP-5/logistic concentrate in R1 (93/95\%, ``true'' European anomalies);
transformers spread into R3 (53--54\%) --- pricing the American wedge itself.
\item \textbf{P3.3} paired exit check: hold-to-expiry beats exit-at-reversion by +\$3.94M (reversion loses money) --- confirms the policy.
\item \textbf{P3.4} beta vs delta: 82\% net-short delta yet positive realized beta; the squared-move (gamma) term
absorbs a large share (MLP-5 +16.5$\to$+8.3) --- short-gamma path-dependence, not a directional bet.
\end{itemize}

\section{Four findings for Pawel (before any paper text change)}
\begin{enumerate}\itemsep2pt
\item \textbf{Commissions are the single dominant friction} (brief-25) --- paper deliberately avoids this attribution.
\item \textbf{The day-selection critique now applies mainly to LightGBM-old} --- the new transformers have genuine
within-day edge (random beats new plain only 8\%, was 62\% for committed plain).
\item \textbf{The new transformers are significantly positive and best-calibrated} (brief-22, calibration).
\item \textbf{Two unreproduced paper numbers} --- logistic Table-2 (13,841/\$2.02M/2.27) and the exact $-$\$0.77M random.
\end{enumerate}

\section{Pipeline re-check and state}
Read-only re-check: all gates pass (new Table 2 reproduces, decomposition net $=-$\$12.52M, calibration
complete, derived-on-old artifacts confirmed fixed to new models). \textbf{Git:} local is ahead of origin with
the rerun work and now includes the merged PI v5 (8 pages) --- \emph{not pushed}. \textbf{Equity figures:}
\code{icaif26/figures/equity\_curves.pdf} (committed-model, matches committed Table 2/3) and
\code{equity\_curves\_newmodel.pdf} (new-model set). \textbf{Read next:} \code{rerun\_outputs\_summary.pdf}
and this file. No v5 tex edited --- insertions await the PI pass.
\end{document}
"""
(HERE/"final_review.tex").write_text(TEX)
print("wrote final_review.tex")
