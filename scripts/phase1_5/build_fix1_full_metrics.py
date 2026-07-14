#!/usr/bin/env python3
"""FIX1 FULL-METRICS report (21-page-style structure) for:
Reward-Gated GNN-LPD Traffic Engineering (RG-GNN-LPD)
formal def: Full-Data-Trained Topology-Agnostic Bottleneck-Ranking DDQN with
Gated First-Cycle Initialization.

All headline numbers come from FIX1 artifacts only. Missing metrics are labeled
honestly and never fabricated. Generates figures into FINAL_REPORT_FIX1/figures/
and writes MISSING_BASELINES.md. Does NOT overwrite the short/comprehensive
reports.
"""
import json, pickle, platform
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

RC = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean/results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50")
SDN = RC.parent / "sdn_mininet_clean"
F1 = RC / "FULLDATA_GATED_PRESERVED_FIX1"; FAIL = RC / "PERIODIC_FAILURE_FIX1"
RPT = RC / "FINAL_REPORT_FIX1"; FIG = RPT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
LABEL = "RG-GNN-LPD"
TOPO = ["abilene","geant","cernet","sprintlink","tiscali","ebone","germany50","vtlwavenet2011"]
DISP = {"abilene":"Abilene","geant":"GEANT","cernet":"CERNET","sprintlink":"Sprintlink","tiscali":"Tiscali","ebone":"Ebone","germany50":"Germany50","vtlwavenet2011":"VtlWavenet2011"}
ZS = {"germany50","vtlwavenet2011"}
EVALWIN = {"abilene":(2016,4032),"geant":(672,1344),"cernet":(200,400),"sprintlink":(200,400),"tiscali":(200,400),"ebone":(200,400),"germany50":(0,288),"vtlwavenet2011":(0,40)}
SRC = {"abilene":"Real / SNDlib","geant":"Real / SNDlib","cernet":"MGM (synthetic-real)","sprintlink":"RocketFuel / MGM","tiscali":"RocketFuel / MGM","ebone":"RocketFuel / MGM","germany50":"Real / SNDlib","vtlwavenet2011":"Topology Zoo / MGM"}

# ---------------- load artifacts ----------------
ev = pd.read_csv(F1 / "fulldata_gated_eval_per_cycle.csv")
agg = pd.read_csv(F1 / "normal_8topo_vs_target.csv").set_index("topology")
adist = pd.read_csv(F1 / "action_distribution.csv").set_index("Topology")
flex = pd.read_csv(F1 / "flexdate_comparison.csv").set_index("topology")
clean = pd.read_csv(F1 / "fix1_clean_timing_per_cycle.csv")
tlog = pd.read_csv(F1 / "finetune_train_log.csv")
counters = json.load(open(F1 / "finetune_counters.json"))
P = pickle.load(open(RC / "_prepass.pkl", "rb"))
GNN_MS = {"abilene":1,"geant":7,"cernet":7,"sprintlink":10,"tiscali":10,"ebone":4,"germany50":26,"vtlwavenet2011":26}
ACTS = ["KEEP","K50","K100","K200","K300","K500","K800"]

# per-cycle MLU from periodic non-failure rows (verified identical PR to eval)
fail = {t: pd.read_csv(FAIL / f"periodic_{t}.csv") for t in TOPO}
def nod_of(t):
    key=(t,)+EVALWIN[t]; d=P.get(key);
    if d is None: return None
    rk=d.get("ranked");
    if isinstance(rk,dict):
        any_t=next(iter(rk)); return len(rk[any_t])
    return None

M = {}   # per-topo metrics
val_notes = []
for t in TOPO:
    g = ev[ev.topology==t].copy()
    lo,hi = EVALWIN[t]; key=(t,lo,hi); d=P.get(key)
    nf = fail[t][fail[t].is_failure==0][["tm_index","MLU"]]
    g = g.merge(nf, on="tm_index", how="left")
    # opt / ecmp per cycle
    optd = d["opt"] if d is not None else {}; emlud = d["emlu"] if d is not None else {}
    g["opt"] = g.tm_index.map(lambda x: optd.get(int(x), np.nan))
    g["emlu"] = g.tm_index.map(lambda x: (emlud[int(x)] if (isinstance(emlud,dict) and int(x) in emlud) else (emlud[int(x)-lo] if hasattr(emlud,"__getitem__") and not isinstance(emlud,dict) and 0<=int(x)-lo<len(emlud) else np.nan)))
    pr = g.PR.values
    mlu = g.MLU.values if "MLU" in g else np.full(len(g),np.nan)
    mlu_ok = mlu[np.isfinite(mlu)]
    n_mlu = int(np.isfinite(mlu).sum())
    # NOTE: eval PR uses the strict full-MCF numerator, while prepass['opt'] is the LP optimum;
    # they differ, so MLU/opt and MLU/ECMP are NOT computed here (would be inconsistent). Marked n/a.
    val_notes.append(f"{t}: MLU rows(non-failure)={n_mlu}/{len(g)}")
    a = agg.loc[t]
    cl = clean[clean.topology==t].decision_ms.values
    M[t] = dict(
        N=len(g), meanPR=pr.mean(), medPR=float(np.median(pr)), minPR=pr.min(),
        pr99=float((pr>=0.99).mean()*100), pr95=float((pr>=0.95).mean()*100), pr90=float((pr>=0.90).mean()*100),
        meanDB=float(a.meanDB), p95DB=float(a.p95DB),
        n_mlu=n_mlu,
        meanMLU=(float(np.mean(mlu_ok)) if n_mlu else None), medMLU=(float(np.median(mlu_ok)) if n_mlu else None),
        p95MLU=(float(np.percentile(mlu_ok,95)) if n_mlu else None), maxMLU=(float(np.max(mlu_ok)) if n_mlu else None),
        meanMLUopt=None, meanMLUecmp=None,  # n/a: numerator/source mismatch (see MISSING_BASELINES.md)
        mean_ms=float(g.decision_ms.mean()), p95_ms=float(np.percentile(g.decision_ms,95)), max_ms=float(g.decision_ms.max()),
        clean_p95=(float(np.percentile(cl,95)) if len(cl) else None),
        nod=nod_of(t),
        adist={k:int(adist.loc[t,k]) for k in ACTS if k in adist.columns},
    )
    M[t]["most_action"] = max(M[t]["adist"], key=M[t]["adist"].get)
print("[validate] " + " | ".join(val_notes))

# ---------------- figures ----------------
plt.rcParams.update({"figure.dpi":130,"font.size":9})
COL = "#2E5496"
def save(fig,name): fig.tight_layout(); fig.savefig(FIG/name, bbox_inches="tight"); plt.close(fig)
# 1 PR CDF (per seen vs zero-shot aggregate)
fig,ax=plt.subplots(figsize=(6,3.4))
for t in TOPO:
    v=np.sort(ev[ev.topology==t].PR.values); ax.plot(v,np.linspace(0,1,len(v)),lw=1.2,label=DISP[t])
ax.set_xlabel("Performance Ratio (PR)"); ax.set_ylabel("CDF"); ax.set_xlim(0.6,1.001); ax.grid(alpha=.3)
ax.legend(fontsize=6,ncol=2); ax.set_title(f"{LABEL}: per-cycle PR CDF (FIX1)"); save(fig,"fig1_pr_cdf.png")
# 3 decision-time CDF
fig,ax=plt.subplots(figsize=(6,3.4))
for t in TOPO:
    v=np.sort(ev[ev.topology==t].decision_ms.values); ax.plot(v,np.linspace(0,1,len(v)),lw=1.2,label=DISP[t])
ax.axvline(500,color="r",ls="--",lw=1,label="500 ms"); ax.set_xlabel("decision time (ms)"); ax.set_ylabel("CDF"); ax.grid(alpha=.3)
ax.legend(fontsize=6,ncol=2); ax.set_title(f"{LABEL}: decision-time CDF (FIX1)"); save(fig,"fig3_ms_cdf.png")
# 4 PR-DB tradeoff (per-topo aggregate)
fig,ax=plt.subplots(figsize=(5.5,3.4))
for t in TOPO:
    ax.scatter(M[t]["meanDB"],M[t]["meanPR"],s=40,color=COL); ax.annotate(DISP[t],(M[t]["meanDB"],M[t]["meanPR"]),fontsize=6,xytext=(3,3),textcoords="offset points")
ax.set_xlabel("mean DB"); ax.set_ylabel("mean PR"); ax.grid(alpha=.3); ax.set_title(f"{LABEL}: PR-DB tradeoff (FIX1)"); save(fig,"fig8_pr_vs_db.png")
# 5 PR>=0.95 bar
fig,ax=plt.subplots(figsize=(6,3.2)); xs=range(len(TOPO))
ax.bar(xs,[M[t]["pr95"] for t in TOPO],color=COL); ax.set_xticks(xs); ax.set_xticklabels([DISP[t] for t in TOPO],rotation=40,ha="right",fontsize=7)
ax.set_ylabel("% cycles PR>=0.95"); ax.set_title(f"{LABEL}: PR>=0.95 by topology (FIX1)"); ax.grid(alpha=.3,axis="y"); save(fig,"fig4_pr95.png")
# 6 mean DB bar
fig,ax=plt.subplots(figsize=(6,3.2))
ax.bar(xs,[M[t]["meanDB"] for t in TOPO],color="#C55A11"); ax.set_xticks(xs); ax.set_xticklabels([DISP[t] for t in TOPO],rotation=40,ha="right",fontsize=7)
ax.set_ylabel("mean DB"); ax.set_title(f"{LABEL}: mean DB by topology (FIX1)"); ax.grid(alpha=.3,axis="y"); save(fig,"fig5_meandb.png")
# 7 action distribution stacked
fig,ax=plt.subplots(figsize=(6.2,3.4)); bottom=np.zeros(len(TOPO)); import matplotlib.cm as cm
cols=plt.cm.viridis(np.linspace(0,1,len(ACTS)))
for j,act in enumerate(ACTS):
    vals=np.array([M[t]["adist"].get(act,0) for t in TOPO],float); tot=np.array([sum(M[t]["adist"].values()) for t in TOPO],float)
    frac=vals/np.where(tot>0,tot,1)*100; ax.bar(xs,frac,bottom=bottom,color=cols[j],label=act); bottom+=frac
ax.set_xticks(list(xs)); ax.set_xticklabels([DISP[t] for t in TOPO],rotation=40,ha="right",fontsize=7); ax.set_ylabel("% of cycles")
ax.legend(fontsize=6,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.18)); ax.set_title(f"{LABEL}: gate action distribution (FIX1)"); save(fig,"fig7_actiondist.png")
# 8 failure MLU CDF (failure cycles)
fig,ax=plt.subplots(figsize=(6,3.4))
for t in TOPO:
    fc=fail[t][fail[t].is_failure==1].MLU.values; fc=np.sort(fc[np.isfinite(fc)])
    if len(fc): ax.plot(fc,np.linspace(0,1,len(fc)),lw=1.2,label=DISP[t])
ax.set_xlabel("MLU on failure cycles"); ax.set_ylabel("CDF"); ax.grid(alpha=.3); ax.legend(fontsize=6,ncol=2)
ax.set_title(f"{LABEL}: failure-cycle MLU CDF (FIX1 stress-test)"); save(fig,"fig9_fail_mlu_cdf.png")
# 9 failure PR by topology (mean + on-failure)
fig,ax=plt.subplots(figsize=(6,3.2)); w=.38
mp=[fail[t].PR.mean() for t in TOPO]; fp=[fail[t][fail[t].is_failure==1].PR.mean() for t in TOPO]
ax.bar([x-w/2 for x in xs],mp,w,label="all cycles",color=COL); ax.bar([x+w/2 for x in xs],fp,w,label="failure cycles",color="#C00000")
ax.set_xticks(list(xs)); ax.set_xticklabels([DISP[t] for t in TOPO],rotation=40,ha="right",fontsize=7); ax.set_ylim(0.9,1.001)
ax.set_ylabel("mean PR"); ax.legend(fontsize=7); ax.set_title(f"{LABEL}: failure-cycle PR (FIX1 stress-test)"); save(fig,"fig10_fail_pr.png")
# 10 mininet throughput/loss (retained)
sdn=pd.read_csv(SDN/"sdn_summary.csv")
fig,ax=plt.subplots(figsize=(6.4,3.2)); scn=sdn[sdn.topology=="abilene"].scenario.tolist()
ab=sdn[sdn.topology=="abilene"]; ge=sdn[sdn.topology=="geant"]
xx=np.arange(len(scn)); ax.bar(xx-0.2,ab.mean_recovery_ms,0.4,label="Abilene recovery ms",color=COL); ax.bar(xx+0.2,ge.mean_recovery_ms,0.4,label="GEANT recovery ms",color="#70AD47")
ax.set_xticks(xx); ax.set_xticklabels(scn,rotation=35,ha="right",fontsize=6); ax.set_ylabel("mean recovery ms")
ax.legend(fontsize=7); ax.set_title("Retained SDN/Mininet recovery (not rerun on FIX1)"); save(fig,"fig11_sdn.png")
# learning curve
fig,ax=plt.subplots(figsize=(5.5,3.2)); ax2=ax.twinx()
ax.plot(tlog.epoch,tlog.mean_reward,"o-",color=COL,label="mean reward"); ax2.plot(tlog.epoch,tlog.mean_distill,"s--",color="#C55A11",label="mean distill loss")
ax.set_xlabel("fine-tune epoch"); ax.set_ylabel("mean reward",color=COL); ax2.set_ylabel("distill loss",color="#C55A11")
ax.set_title(f"{LABEL}: FIX1 fine-tune curve"); ax.set_xticks(tlog.epoch); save(fig,"fig_learn.png")
print("[figures] written to", FIG)

# ---------------- MISSING_BASELINES.md ----------------
(RPT/"MISSING_BASELINES.md").write_text(
"# Missing baselines / experiments for FIX1 (RG-GNN-LPD)\n\n"
"The following were NOT re-run under the FIX1 model + protocol. They must be generated before the\n"
"corresponding rows can be filled with FIX1 numbers. Do not copy Frozen Tier A numbers as FIX1.\n\n"
"## Baselines not run for FIX1\n"
"- OSPF / shortest-path MLU per cycle\n- Top-K demand selection\n- Bottleneck-critical Top-K (standalone)\n"
"- GNN-only fixed K30 / LPD-only fixed K30\n- GNN+LPD fixed K30 / K40 / K50 (global, all topologies)\n"
"- GNN-only + reward gate / LPD-only + reward gate\n\n"
"## Available real reference points for FIX1 (used in the report)\n"
"- ECMP per-cycle MLU reference: from _prepass.pkl 'emlu' (background reference only).\n"
"- Full-MCF / LP optimum per-cycle MLU reference: from _prepass.pkl 'opt' (used only through PR = optimum/achieved).\n"
"- These cached references are NOT used to report MLU/optimum or MLU/ECMP ratios in the FIX1 report because\n"
"  the numerator basis is not safely aligned with the strict headline PR computation.\n\n"
"## K-sensitivity not run for FIX1\n"
"- Global GNN+LPD fixed K10..K50 sweep (FIX1). Only a VtlWavenet forced-K frontier exists (vtl_scalability.csv,\n"
"  measured on the iter2 frozen model) and is labeled historical in the report.\n\n"
"## Failure scenarios not run for FIX1 in the LP-eval harness\n"
"- two-link / three-link / capacity-degradation-50 / mixed spike+failure at the LP-eval level.\n"
"- These exist only in the retained SDN/Mininet artifact (sdn_summary.csv, Abilene+GEANT), NOT rerun on FIX1.\n"
"- FIX1 has the periodic single-link failure stress-test (PERIODIC_FAILURE_FIX1/).\n\n"
"## Instrumentation gaps\n"
"- Per-cycle realized DB is not logged (per-cycle DB column stores the budget); only aggregate mean/p95 DB exist.\n"
"  => median DB, max DB, and a DB CDF cannot be computed without re-running the eval with per-cycle DB logging.\n"
"- Decision-time components (GNN vs LPD vs fusion vs gate vs LP vs DB calc) are not separately instrumented;\n"
"  only total decision_ms and the constant GNN inference offset are known (2-way split only).\n")
print("[missing] MISSING_BASELINES.md written")

json.dump({k:{kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk!="adist"} for k,v in M.items()},
          open(RPT/"fix1_full_metrics.json","w"), indent=2)
print("DONE_METRICS")
