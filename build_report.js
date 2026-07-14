const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType, PageBreak,
} = require("docx");

const D = JSON.parse(fs.readFileSync("/tmp/report_data.json", "utf8"));
const ACCENT = "1F4E79", HEADER_FILL = "D5E8F0", WINF = "E2EFDA", LOSSF = "FCE4E4";
const CW = 9360;

const border = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, { bold = false, fill = null, align = AlignmentType.LEFT, w = null, color = null } = {}) {
  return new TableCell({
    borders,
    width: w ? { size: w, type: WidthType.DXA } : undefined,
    shading: fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text: String(text), bold, size: 18, color: color || "000000", font: "Arial" })] })],
  });
}
function table(headers, rows, widths, rowFill = null) {
  const head = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, { bold: true, fill: HEADER_FILL, align: AlignmentType.CENTER, w: widths[i] })) });
  const body = rows.map((r) => new TableRow({
    children: r.map((v, i) => {
      let f = null;
      if (rowFill) f = rowFill(r);
      return cell(v, { fill: f, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, w: widths[i] });
    }),
  }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })] });
const P = (t, { bold = false, italics = false } = {}) => new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: t, bold, italics, size: 21, font: "Arial" })] });
const bullet = (t) => new Paragraph({ bullet: { level: 0 }, spacing: { after: 40 }, children: [new TextRun({ text: t, size: 21, font: "Arial" })] });
const winFill = (r) => { const s = r[r.length - 1]; return s === "WIN" ? WINF : (s === "loss" ? LOSSF : null); };

const kids = [];
// Title
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1400, after: 120 }, children: [new TextRun({ text: "GNN-LPD-DQN Selective-Flow Traffic Engineering", bold: true, size: 40, color: ACCENT, font: "Arial" })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Official Evaluation Report", bold: true, size: 30, color: "404040", font: "Arial" })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun({ text: "Strict K10–K50 condition-compliant method + emergency-expanded selected-OD tier", italics: true, size: 22, color: "606060", font: "Arial" })] }));
kids.push(new Paragraph({ children: [new PageBreak()] }));

// Executive summary
kids.push(H1("Executive Summary"));
kids.push(P("This report evaluates a GNN-LPD-DQN selective-flow traffic-engineering method on six seen and two zero-shot topologies, under FlexDATE comparison targets, link-failure robustness, and an SDN/Mininet operational validation. Two tracks are reported and kept separate:"));
kids.push(bullet("Track A — strict condition-compliant method: a Double-DQN selects one of seven actions (KEEP, OPTIMIZE_K10/K20/K30/K40/K50, EMERGENCY) using literal OD counts (K ≤ 50). The LP optimizes only the selected top-K OD pairs; all nonselected OD pairs remain on ECMP."));
kids.push(bullet("Track B — emergency-expanded selected-OD tier: the EMERGENCY action may use a percentage budget (selected_K = min(cap, ⌈pct%·active⌉)) with dynamic path-subset selection from the fixed 8-path library. It remains selected-OD (GNN-LPD ranks ODs, the LP optimizes only the selected set, nonselected ODs stay ECMP, all_od_lp_used = 0)."));
kids.push(P("Headline: Track A wins FlexDATE PR+DB on 3/5 source-locked topologies (Abilene, CERNET, GEANT). Track B additionally recovers Sprintlink to a PR+DB win at sub-500 ms decision time, giving 4/5.", { bold: true }));

// Architecture & pipeline
kids.push(H1("1. Method Architecture and Pipeline"));
kids.push(P("The method is a learned controller (Double-DQN) over an optimization actuator (selected-flow LP), with a GNN-LPD criticality scorer for OD ranking. Per cycle:"));
[["1.", "Static topology + dynamic traffic matrix; 8 candidate paths per OD; ECMP baseline."],
 ["2.", "GNN-LPD scorer ranks all active OD pairs by criticality (LP-distilled GraphSAGE)."],
 ["3.", "Double-DQN reads deployable state features and selects the action (the K budget)."],
 ["4.", "The selected-flow DB-budgeted LP optimizes only the top-K ranked OD pairs (single solve, K ≤ 50 in Track A)."],
 ["5.", "All nonselected OD pairs remain on ECMP (audited: num_non_ecmp ≤ 50 in Track A)."],
 ["6.", "Routing is committed; PR, DB and decision time are recorded."]].forEach(([n, t]) => kids.push(new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: n + " ", bold: true, size: 21, font: "Arial" }), new TextRun({ text: t, size: 21, font: "Arial" })] })));
kids.push(P("An OD pair is one source–destination traffic demand. The GNN-LPD scorer and the Double-DQN are the only learned components; the routing itself is computed by the LP. The DQN genuinely controls the action each cycle (removing it changes the routing).", { italics: true }));

// Action space
kids.push(H2("1.1 Action space (seven actions)"));
kids.push(table(["Action", "K (top-ranked ODs optimized)", "Notes"],
  [["KEEP_PREVIOUS_ROUTING", "0", "reuse last committed routing"],
   ["OPTIMIZE_K10 / K20 / K30", "10 / 20 / 30", "normal selected-flow budgets"],
   ["OPTIMIZE_K40 / K50", "40 / 50", "max normal budget"],
   ["EMERGENCY", "50 (Track A) ; % budget (Track B)", "stronger DB-budget repair; expanded in Track B"]],
  [3000, 3600, 2760]));

// Dataset protocol
kids.push(H1("2. Dataset and Evaluation Protocol"));
kids.push(bullet("Seen training topologies: Abilene, GÉANT, CERNET, Sprintlink, Tiscali, Ebone."));
kids.push(bullet("Zero-shot topologies (no retraining): Germany50, VtlWavenet2011."));
kids.push(bullet("Traffic matrices: Abilene 2016 train + 2016 test; GEANT 672 + 672; CERNET/Rocketfuel/MGM 200 + 200 synthetic; Germany50 real TMs for generalization."));
kids.push(bullet("FlexDATE PR/DB targets per topology are used as the comparison bar."));

// All-topology results
kids.push(H1("3. Results — All Topologies (Track A, strict K10–K50)"));
kids.push(table(["Topology", "N", "PR", "DB", "Mean ms", "P95 ms"],
  D.all_topo.map((r) => [r[0], r[1], r[2].toFixed(4), r[3].toFixed(4), r[4], r[5]]),
  [2400, 1100, 1600, 1600, 1330, 1330]));
kids.push(P("Decision time = full per-cycle decision (GNN scoring + DQN + LP). A separate runtime column is not recorded; runtime equals decision time.", { italics: true }));

// FlexDATE Track A
kids.push(H1("4. FlexDATE Comparison — Track A (strict K10–K50)"));
kids.push(table(["Topology", "Our PR", "FlexDATE PR", "Our DB", "FlexDATE DB", "Mean ms", "P95 ms", "Winner"],
  D.trackA.map((r) => [r[0], r[1].toFixed(4), r[2], r[3].toFixed(4), r[4], r[5], r[6], r[7] === "WIN" ? "OURS" : "FlexDATE"]),
  [1700, 1150, 1250, 1150, 1250, 980, 980, 900], (r) => (r[7] === "OURS" ? WINF : LOSSF)));
kids.push(P("Track A: 3/5 FlexDATE PR+DB wins (Abilene, CERNET, GEANT), every topology under 110 ms. Sprintlink and Tiscali lose PR because K ≤ 50 optimizes at most 50 of their ~1900–2350 OD pairs.", { bold: true }));

// FlexDATE Track B
kids.push(H1("5. FlexDATE Comparison — Track B (emergency-expanded selected-OD)"));
kids.push(table(["Topology", "Our PR", "FlexDATE PR", "Our DB", "FlexDATE DB", "Mean ms", "P95 ms", "Mode"],
  D.trackB.map((r) => [r[0], r[1].toFixed(4), r[2], r[3].toFixed(4), r[4], r[5], r[6], r[8]]),
  [1500, 1080, 1100, 1080, 1100, 900, 900, 1700],
  (r) => (parseFloat(r[1]) >= parseFloat(r[2]) && parseFloat(r[3]) < parseFloat(r[4]) ? WINF : LOSSF)));
kids.push(P("Track B: 4/5 FlexDATE PR+DB wins. Sprintlink is recovered by the EMERGENCY percentage-budget tier (75% → 1400 selected ODs, 3 of the 8 candidate paths used), reaching PR 0.9991, DB 0.0006 at 270 ms mean / 382 ms P95 — both under 500 ms. This is reported as emergency-expanded selected-OD (all_od_lp_used = 0, nonselected = ECMP), not as the strict K ≤ 50 result.", { bold: true }));

// Old report comparison
kids.push(H1("6. Comparison with Old Report"));
kids.push(P("The old report's K30/K40/K50 labels denote percentage budgets (30/40/50% of active ODs, capped at 800), i.e. hundreds of OD pairs — not literal counts. The strict K ≤ 50 method is therefore not expected to reproduce the old report."));
kids.push(table(["Topology", "Our PR", "Old PR", "PR", "Our DB", "Old DB", "DB", "Both"],
  D.oldcmp.map((r) => [r[0], r[1].toFixed(4), r[2].toFixed(4), r[3], r[4].toFixed(4), r[5].toFixed(4), r[6], r[7]]),
  [1700, 1150, 1150, 800, 1150, 1150, 800, 1460],
  (r) => (r[7] === "WIN" ? WINF : null)));
kids.push(P("Versus the old report: CERNET wins both metrics; DB is superior on 3/5 (CERNET, Sprintlink 0.0006 vs 0.0319, Tiscali 0.0007 vs 0.0170); PR is competitive (GEANT win, Abilene −0.0003). The old report's ultra-low Abilene/GEANT DB and exact Sprintlink PR = 1.0 stem from its percentage-budget design and disturbance-finalization, which the strict conditions exclude."));

// Failures
kids.push(H1("7. Failure Robustness"));
kids.push(P("Nine scenarios (normal + eight failure/stress cases) were evaluated on Abilene and GÉANT. Failure-aware routing prunes dead-link paths so the LP/ECMP reroute around failures (overall mean PR 0.901; finite MLU)."));
kids.push(table(["Topology", "Scenario", "Mean PR", "Mean MLU", "Mean DB", "Mean ms", "Disc. ODs"],
  D.fail.map((r) => [r[0], r[1], r[2].toFixed(3), r[3].toFixed(3), r[4].toFixed(4), r[5], r[6]]),
  [1300, 2700, 1200, 1200, 1200, 980, 780]));

// SDN
kids.push(H1("8. SDN / Mininet Operational Validation"));
kids.push(P("The GNN-LPD-DQN routing solutions were evaluated under operational scenarios for Abilene and GÉANT (LP-simulation mode; Mininet not available on macOS — the script supports --mode mininet on Linux/OVS). Metrics:"));
kids.push(table(["Topology", "Scenario", "Thr (Mbps)", "Loss %", "RTT ms", "Recovery ms", "Flow rules", "Install ms"],
  D.sdn.map((r) => [r[0], r[1], r[2], r[3].toFixed(1), r[4].toFixed(1), r[5].toFixed(1), r[6], r[7].toFixed(2)]),
  [1200, 2400, 1120, 820, 980, 1320, 1100, 1420]));

// Compliance
kids.push(H1("9. Compliance Audit"));
kids.push(bullet("Track A (strict): action set = {KEEP, OPTIMIZE_K10..K50, EMERGENCY}; max_selected_K = 50; num_non_ecmp ≤ 50; full_od_lp_used = 0; hidden_k_escalation = 0; nonselected = ECMP; uses_optimal_at_inference = false; deployable = true; RF not used. PASS."));
kids.push(bullet("Track B (emergency-expanded): action = EMERGENCY; emergency_budget_type = percentage; selected_od_lp_used = 1; all_od_lp_used = 0; nonselected = ECMP; gnn_lpd_used = 1; dqn_used_at_inference = 1; exceeds_literal_K50_condition = true (disclosed)."));

// Limitations & reproducibility
kids.push(H1("10. Limitations and Reproducibility"));
kids.push(bullet("Under strict K ≤ 50, Sprintlink and Tiscali cannot reach FlexDATE PR (0.999): their bottlenecks span far more than 50 OD pairs (Sprintlink needs ≈ 1400 ODs ≈ 75%)."));
kids.push(bullet("The emergency tier recovers Sprintlink within 500 ms via path-subset selection; Tiscali was not yet recovered."));
kids.push(bullet("SDN metrics are LP-simulation values; real Mininet requires Linux/OVS."));
kids.push(bullet("All results come from a single official pipeline (gnn_lpd_dqn_selective_db_lp). Strict and emergency tracks use the same scorer, references and 8-path library."));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, color: ACCENT, font: "Arial" }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, color: "2E75B6", font: "Arial" }, paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: "bullet", text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children: kids }],
});
Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync("results/gnn_lpd_dqn_selective_db_lp/OFFICIAL_EVALUATION_REPORT.docx", b);
  console.log("WROTE OFFICIAL_EVALUATION_REPORT.docx", b.length, "bytes");
});
