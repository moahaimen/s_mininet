#!/usr/bin/env python3
"""Method-consistent FAILURE-AWARE fine-tune of the FIX1 DDQN.
- Failure states: regress Q(state,:) toward the precomputed per-action reward vector (terminal targets) so the
  DDQN LEARNS to pick the higher-value optimize action under failure. DDQN remains the action selector; no
  heuristic override, no forced K800, same authorized action space, same reward as FIX1.
- Normal states: distillation (MSE of Q to the frozen FIX1 teacher) to PRESERVE the N=3976 normal behavior.
Failure states are feature-separable (high keep_mlu), so failure Q can change while normal Q is anchored to FIX1.
Output: FAILAWARE_FINETUNE/failaware_model.pt (does NOT overwrite FIX1)."""
import sys, pickle, json, random, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT, set_seed
set_seed(42); random.seed(42); np.random.seed(42); torch.manual_seed(42)
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "FAILAWARE_FINETUNE"; FC = OUT / "RETRAIN_PHASE1_FULLTRAIN" / "_cache_full"
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
dim = len(A.AGN_FEAT_NAMES); FIX1 = OUT / "FULLDATA_GATED_PRESERVED_FIX1" / "fulldata_gated_model.pt"
def stf(raw, keep_mlu, emlu): return A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD).astype(np.float32)

# ---- models ----
teacher = A.QNet(dim, 7); teacher.load_state_dict(torch.load(FIX1, map_location="cpu")["state_dict"]); teacher.eval()
online = A.QNet(dim, 7); online.load_state_dict(teacher.state_dict())

# ---- normal distillation anchor states (FIX1 training states; static keep~emlu, like action_match) ----
SEEN = ["abilene", "geant", "cernet", "sprintlink", "tiscali", "ebone"]
normal_states = []
for topo in SEEN:
    raws = pickle.load(open(FC / f"raw_{topo}.pkl", "rb"))
    for t, (raw, emlu) in raws.items():
        normal_states.append(stf(raw, emlu, emlu))
normal_states = np.array(normal_states, np.float32)
print(f"[data] normal distill states: {len(normal_states)}", flush=True)

# ---- failure experiences (state + per-action reward) ----
fexp = pickle.load(open(SUB / "failure_experiences.pkl", "rb"))
fail_states = np.array([e["state"] for e in fexp], np.float32)
fail_rewards = np.array([e["reward"] for e in fexp], np.float32)
print(f"[data] failure experiences: {len(fexp)} ({fail_states.shape})", flush=True)

# ---- normalize reward targets to the teacher-Q scale (per-state offset+scale preserves argmax) ----
# regress failure Q toward reward, but shift/scale rewards into a comparable numeric range for stable training.
with torch.no_grad():
    tqf = teacher(torch.tensor(fail_states)).numpy()            # FIX1 Q on failure states (starting point)
rmin = fail_rewards.min(axis=1, keepdims=True); rmax = fail_rewards.max(axis=1, keepdims=True)
qmin = tqf.min(axis=1, keepdims=True); qmax = tqf.max(axis=1, keepdims=True)
scale = (qmax - qmin) / np.maximum(rmax - rmin, 1e-6)
fail_targets = (qmin + (fail_rewards - rmin) * scale).astype(np.float32)  # same ranking as reward, teacher-Q scale

opt = torch.optim.Adam(online.parameters(), 2e-4); huber = nn.SmoothL1Loss(); mse = nn.MSELoss()
LAMBDA_DISTILL = 5.0; EPOCHS = 20; BATCH = 256
NS = torch.tensor(normal_states); FS = torch.tensor(fail_states); FT = torch.tensor(fail_targets)
n_f = len(fexp)
log = []
for ep in range(EPOCHS):
    perm = torch.randperm(n_f); tds = []; dls = []
    for i in range(0, n_f, BATCH):
        idx = perm[i:i+BATCH]
        qf = online(FS[idx]); td = huber(qf, FT[idx])                    # learn failure ranking
        nidx = torch.randint(0, len(normal_states), (min(BATCH, len(normal_states)),))
        with torch.no_grad(): tqn = teacher(NS[nidx])
        dl = mse(online(NS[nidx]), tqn)                                  # preserve normal
        loss = td + LAMBDA_DISTILL * dl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
        tds.append(float(td)); dls.append(float(dl))
    # diagnostics: fraction of failure states where argmax action >= K100 (i.e., no longer KEEP/K50)
    with torch.no_grad():
        af = online(FS).argmax(1).numpy()
        agree_norm = (online(NS).argmax(1) == teacher(NS).argmax(1)).float().mean().item()
    hi_frac = float(np.mean(af >= 2))  # actions 2..6 = K100..K800
    log.append(dict(epoch=ep+1, td=round(np.mean(tds), 4), distill=round(np.mean(dls), 5),
                    fail_highK_frac=round(hi_frac, 3), normal_action_match=round(agree_norm, 4)))
    print(f"  ep{ep+1:2d} td={np.mean(tds):.4f} distill={np.mean(dls):.5f} | fail argmax>=K100: {hi_frac*100:.0f}% | normal action-match(FIX1): {agree_norm*100:.1f}%", flush=True)

torch.save({"state_dict": online.state_dict(), "dim": dim, "n_act": 7,
            "claim": "Failure-aware fine-tune of FIX1: DDQN learns failure action selection via reward regression on "
                     "failure states + distillation on normal states. DDQN still selects; no heuristic override.",
            "from_scratch": False, "fine_tuned_from": "fulldata_gated_model.pt"}, SUB / "failaware_model.pt")
json.dump(log, open(SUB / "failaware_train_log.json", "w"), indent=2)
print("[saved] failaware_model.pt"); print("DONE")
