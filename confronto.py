#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
confronto.py -- Confronto tra run LLM e benchmark manuale (Output a MANO.xlsx)
======================================================================================
Legge results.jsonl di una o più cartelle-run (schema v1 di newex.py, con `cup` stringa,
oppure schema v2 di mcex.py, con `cup` lista) e li confronta con il foglio Excel della
Fase alpha (colonne DOCUMENTO / CUP / IMPORTO: una riga per coppia CUP-importo; facoltative
CAPITOLO/PG e TIPOLOGIA, usate da punteggio.py dove compilate).

CONVENZIONI DI CONFRONTO (vanno lette insieme ai numeri):
  - Il benchmark manuale RIPARTISCE IN PARTI UGUALI un importo riferito a più CUP
    (es. 50.000.000 su due lotti -> 25.000.000 per CUP). Il prompt v2 (regola R6) chiede
    invece al modello di NON ripartire e di restituire l'importo intero con la lista dei CUP.
    Per confrontare con il benchmark, ogni record v2 viene quindi esploso per CUP con
    importo_quota = importo / n_cup. Con lo schema v1 (un CUP per record) la quota coincide
    con l'importo.
  - Livelli di confronto:
      pair  : insieme di (documento, CUP, importo arrotondato all'euro), in valore assoluto.
      cup   : insieme di (documento, CUP).
      total : somma per (documento, CUP) — per v2 solo i record livello_importo="totale"
              (le quote replicherebbero il totale); tollera 1 euro di scarto.
    "pair" è la misura più severa (granularità identica al benchmark), "total" la più
    robusta alle differenze di granularità (annualità, piani gestionali).
  - I documenti del corpus SENZA coppie nel benchmark (8 su 25) concorrono solo alla
    precisione: ogni coppia estratta lì è un falso positivo.

USO:
    python confronto.py <run_A> [<run_B>]  [--gt "Output a MANO.xlsx"] [--tex out.tex]
    (run = nome cartella sotto output/ oppure percorso)
Scrive confronto_<run>.csv (per modello) e confronto_<run>_docs.csv (per documento) in
ogni cartella-run; con --tex scrive le tabelle LaTeX (booktabs) del confronto A vs B.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
GT_DEFAULT = ROOT / "Output a MANO.xlsx"


# ----------------------------------------------------------------------------- normalizzazione
def _cifre(v) -> str | None:
    """Solo le cifre, senza zeri iniziali: "PG 02" -> "2", "7419" -> "7419"."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = "".join(ch for ch in str(v) if ch.isdigit())
    return str(int(s)) if s else None


def norm_cap_pg(capitolo, pg=None) -> str | None:
    """Chiave "capitolo/pg" comune a benchmark e modello.

    Il benchmark scrive "7419/2" in una sola colonna; il modello restituisce capitolo e
    piano_gestionale separati, ma a volte scrive tutto nel campo capitolo ("7400/PG 7").
    Senza PG la chiave resta incompleta ("7419/") e quindi non combacia: e' voluto, il
    ponte PG -> CUP e' proprio cio' che si misura."""
    if capitolo is None or (isinstance(capitolo, float) and math.isnan(capitolo)):
        return None
    s = str(capitolo).strip()
    if pg is None and "/" in s:
        s, pg = s.split("/", 1)
    cap = _cifre(s)
    if cap is None:
        return None
    return f"{cap}/{_cifre(pg) or ''}"


# Radice -> valore del vocabolario chiuso di stato_importo (mcex.py, sezione 1). Serve ad
# assorbire le varianti che si scrivono a mano ("impegnate", "Assegnato", "revoca").
_TIPOLOGIE = {"impeg": "impegnato", "asseg": "assegnato", "rimod": "rimodulato",
              "revoc": "revocato", "ammes": "ammesso", "ammis": "ammissibile",
              "econo": "economia", "non_a": "non_allocato", "non a": "non_allocato"}


def norm_tipologia(v) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().lower()
    return _TIPOLOGIE.get(s[:5], s) if s else None


# ----------------------------------------------------------------------------- benchmark
def load_gt(path: Path) -> pd.DataFrame:
    """Colonne obbligatorie DOCUMENTO / CUP / IMPORTO. Facoltative CAPITOLO/PG e TIPOLOGIA:
    dove sono compilate il documento si valuta al livello pieno (vedi punteggio.py)."""
    gt = pd.read_excel(path)
    gt = gt.rename(columns={c: c.strip().upper() for c in gt.columns})
    gt = gt.dropna(subset=["DOCUMENTO", "CUP", "IMPORTO"])
    gt["doc"] = gt["DOCUMENTO"].astype(str).str.strip()
    gt["cup"] = gt["CUP"].astype(str).str.strip().str.upper()
    gt["imp"] = gt["IMPORTO"].astype(float).round(0).abs()
    col_pg = next((c for c in ("CAPITOLO/PG", "CAPITOLO_PG", "CAP/PG") if c in gt.columns), None)
    gt["cap_pg"] = gt[col_pg].map(norm_cap_pg) if col_pg else None
    gt["tipologia"] = gt["TIPOLOGIA"].map(norm_tipologia) if "TIPOLOGIA" in gt.columns else None
    return gt[["doc", "cup", "imp", "cap_pg", "tipologia"]]


# ----------------------------------------------------------------------------- run
def load_run(run: str) -> tuple[Path, list[dict]]:
    p = Path(run)
    if not p.exists():
        p = OUTPUT_DIR / run
    jl = p / "results.jsonl"
    if not jl.exists():
        sys.exit(f"results.jsonl non trovato in {p} (usa --export-only per rigenerarlo)")
    return p, [json.loads(l) for l in jl.open(encoding="utf-8") if l.strip()]


def doc_key(decree_file: str) -> str:
    """Chiave documento = stem del PDF. UNICA definizione: prima veniva ricavata in due modi
    diversi (con e senza controllo dell'estensione), e le due tabelle potevano disallinearsi."""
    s = str(decree_file)
    return s[:-4] if s.lower().endswith(".pdf") else s


def explode_records(entry: dict) -> list[dict]:
    """Un dizionario per (record x CUP), con importo intero e quota. Gestisce v1 e v2."""
    out = []
    doc = doc_key(entry["decree_file"])
    for r in entry.get("records") or []:
        cups = r.get("cup")
        cups = [cups] if isinstance(cups, str) else list(cups or [])
        cups = [c for c in cups if c]
        # Il prompt tiene CUP provvisori e CUP Master come valori distinti dal CUP ordinario
        # (mcex.py, sezione 1). Il benchmark manuale invece li registra nella colonna CUP: se
        # non li si raccoglie qui, quei record restano invisibili al confronto e il documento
        # diventa insolubile per costruzione, non per un errore del modello.
        if not cups:
            for campo in ("cup_provvisorio", "cup_master"):
                v = r.get(campo)
                if isinstance(v, str) and v.strip():
                    cups = [v.strip()]
                    break
        imp = r.get("importo_eur")
        if imp is None or not cups:
            continue
        imp = abs(float(imp))
        for c in cups:
            out.append({
                "doc": doc, "cup": str(c).strip().upper(),
                "imp_full": round(imp, 0), "imp_quota": round(imp / len(cups), 0),
                "livello": r.get("livello_importo") or "totale",
                "stato": r.get("stato_importo"), "conf": r.get("confidenza"),
                "cap_pg": norm_cap_pg(r.get("capitolo"), r.get("piano_gestionale")),
                "tipologia": norm_tipologia(r.get("stato_importo")),
                "n_cup": len(cups),
            })
    return out


def prf(tp: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_model(entries: list[dict], gt: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    gt_pairs = set(zip(gt.doc, gt.cup, gt.imp))
    gt_cups = set(zip(gt.doc, gt.cup))
    gt_tot = gt.groupby(["doc", "cup"])["imp"].sum().to_dict()

    rows = [x for e in entries for x in explode_records(e)]
    pred_pairs = {(x["doc"], x["cup"], x["imp_quota"]) for x in rows}
    pred_pairs_full = {(x["doc"], x["cup"], x["imp_full"]) for x in rows}
    pred_pairs_tot = {(x["doc"], x["cup"], x["imp_quota"]) for x in rows if x["livello"] == "totale"}
    pred_cups = {(x["doc"], x["cup"]) for x in rows}
    tot = defaultdict(float)
    for x in rows:
        if x["livello"] == "totale":
            tot[(x["doc"], x["cup"])] += x["imp_quota"]
    tot_ok = sum(1 for k, v in tot.items() if k in gt_tot and abs(gt_tot[k] - v) <= 1.0)

    P, R, F = prf(len(pred_pairs & gt_pairs), len(pred_pairs), len(gt_pairs))
    Pf, Rf, Ff = prf(len(pred_pairs_full & gt_pairs), len(pred_pairs_full), len(gt_pairs))
    Pq, Rq, Fq = prf(len(pred_pairs_tot & gt_pairs), len(pred_pairs_tot), len(gt_pairs))
    Pc, Rc, Fc = prf(len(pred_cups & gt_cups), len(pred_cups), len(gt_cups))
    Pt, Rt, Ft = prf(tot_ok, len(tot), len(gt_tot))

    status = pd.Series([e.get("status") for e in entries]).value_counts().to_dict()
    cost = sum((e.get("cost_usd_api") or 0) for e in entries)
    lat = [e.get("latency_seconds") for e in entries if e.get("latency_seconds") is not None]
    ptok = sum((e.get("prompt_tokens") or 0) for e in entries)
    ctok = sum((e.get("completion_tokens") or 0) for e in entries)
    n_rec = sum(len(e.get("records") or []) for e in entries)
    n_agg = sum(1 for x in rows if x["n_cup"] > 1)
    conf = pd.Series([x["conf"] for x in rows]).value_counts(dropna=False).to_dict()
    stato = pd.Series([x["stato"] for x in rows]).value_counts(dropna=False).to_dict()

    summary = {
        "n_cells": len(entries), "status": json.dumps(status, ensure_ascii=False),
        "n_records": n_rec, "n_pairs_pred": len(pred_pairs), "n_pairs_gt": len(gt_pairs),
        "pair_P": P, "pair_R": R, "pair_F1": F,
        "pairfull_P": Pf, "pairfull_R": Rf, "pairfull_F1": Ff,
        "pairtot_P": Pq, "pairtot_R": Rq, "pairtot_F1": Fq, "n_pairs_tot": len(pred_pairs_tot),
        "cup_P": Pc, "cup_R": Rc, "cup_F1": Fc,
        "total_P": Pt, "total_R": Rt, "total_F1": Ft,
        "cup_rows_aggregated": n_agg,
        "cost_usd": cost, "prompt_tokens": ptok, "completion_tokens": ctok,
        "latency_mean_s": (sum(lat) / len(lat)) if lat else None,
        "confidenza": json.dumps(conf, ensure_ascii=False), "stato_importo": json.dumps(stato, ensure_ascii=False),
    }

    # per documento
    docs = sorted({doc_key(e["decree_file"]) for e in entries} | set(gt.doc))
    drows = []
    for d in docs:
        g = {k for k in gt_pairs if k[0] == d}
        p = {k for k in pred_pairs_tot if k[0] == d}
        st = [e.get("status") for e in entries if doc_key(e["decree_file"]) == d]
        drows.append({"doc": d, "status": st[0] if st else "n/d", "gt": len(g), "pred": len(p),
                      "tp": len(p & g), "fn": len(g - p), "fp": len(p - g)})
    return summary, pd.DataFrame(drows)


def score_run(run: str, gt: pd.DataFrame) -> tuple[Path, pd.DataFrame, dict[str, pd.DataFrame]]:
    run_dir, entries = load_run(run)
    by_model = defaultdict(list)
    for e in entries:
        by_model[f"{e['model_slug']}::{e['thinking']}"].append(e)
    rows, docs = [], {}
    for cell, ents in sorted(by_model.items()):
        s, d = score_model(ents, gt)
        rows.append({"run": run_dir.name, "cell": cell, **s})
        docs[cell] = d
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"confronto_{run_dir.name}.csv", index=False, encoding="utf-8-sig")
    pd.concat([d.assign(cell=c) for c, d in docs.items()]).to_csv(
        run_dir / f"confronto_{run_dir.name}_docs.csv", index=False, encoding="utf-8-sig")
    return run_dir, df, docs


# ----------------------------------------------------------------------------- output
METRICS = [("pair_P", "P"), ("pair_R", "R"), ("pair_F1", "F1"),
           ("cup_F1", "F1 CUP"), ("total_F1", "F1 tot."), ("cost_usd", "USD")]


def print_summary(df: pd.DataFrame) -> None:
    cols = ["run", "cell", "n_records", "n_pairs_pred", "pair_P", "pair_R", "pair_F1",
            "pairtot_P", "pairtot_R", "pairtot_F1", "cup_F1", "total_F1", "cost_usd", "latency_mean_s", "status"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(df[cols].to_string(index=False))


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def tex_tables(dfA: pd.DataFrame, dfB: pd.DataFrame, docsA: dict, docsB: dict, out: Path) -> None:
    """Due tabelle booktabs: (1) metriche per modello, run A vs B; (2) per documento,
    falsi negativi/positivi a livello 'pair' per il modello comune migliore."""
    def short(cell: str) -> str:
        return cell.split("/")[-1].replace("::off", "").replace("::on", " (thinking)")
    lines = []
    lines.append("% Tabella generata da confronto.py")
    lines.append(r"\begin{table}[htbp]\centering\small")
    lines.append(r"\caption{Confronto tra prompt v1 (run di riferimento) e prompt v2 (mcex.py) sul benchmark manuale}")
    lines.append(r"\label{tab:confronto_prompt}")
    lines.append(r"\begin{tabular}{llrrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & & & \multicolumn{3}{c}{Coppie (tutti i record)} & \multicolumn{3}{c}{Coppie (solo totali)} & & \\")
    lines.append(r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}")
    lines.append(r"Modello & Prompt & Record & P & R & F1 & P & R & F1 & F1$_{\text{CUP}}$ & Costo (USD) \\")
    lines.append(r"\midrule")
    cells = sorted(set(dfA.cell) | set(dfB.cell))
    for c in cells:
        for label, df in (("v1", dfA), ("v2", dfB)):
            r = df[df.cell == c]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"{tex_escape(short(c))} & {label} & {int(r.n_records)} & {r.pair_P:.2f} & {r.pair_R:.2f} & "
                         f"{r.pair_F1:.2f} & {r.pairtot_P:.2f} & {r.pairtot_R:.2f} & {r.pairtot_F1:.2f} & "
                         f"{r.cup_F1:.2f} & {r.cost_usd:.2f} \\\\")
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    # tabella per documento sul modello comune con F1 v2 più alto
    common = [c for c in cells if c in set(dfA.cell) and c in set(dfB.cell)]
    if common:
        best = max(common, key=lambda c: float(dfB[dfB.cell == c].pair_F1.iloc[0]))
        dA, dB = docsA[best].set_index("doc"), docsB[best].set_index("doc")
        lines.append(r"\begin{table}[htbp]\centering\small")
        lines.append(r"\caption{Dettaglio per documento (coppie CUP--importo, soli record di livello totale), modello " + tex_escape(short(best)) + "}")
        lines.append(r"\label{tab:confronto_doc}")
        lines.append(r"\begin{tabular}{lrrrrrrr}")
        lines.append(r"\toprule")
        lines.append(r" & & \multicolumn{3}{c}{Prompt v1} & \multicolumn{3}{c}{Prompt v2} \\")
        lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
        lines.append(r"Documento & Bench. & TP & FN & FP & TP & FN & FP \\")
        lines.append(r"\midrule")
        for d in sorted(set(dA.index) | set(dB.index)):
            a = dA.loc[d] if d in dA.index else None
            b = dB.loc[d] if d in dB.index else None
            g = int((a if a is not None else b)["gt"])
            fa = (f"{int(a.tp)} & {int(a.fn)} & {int(a.fp)}" if a is not None else "-- & -- & --")
            fb = (f"{int(b.tp)} & {int(b.fn)} & {int(b.fp)}" if b is not None else "-- & -- & --")
            lines.append(f"{tex_escape(d)} & {g} & {fa} & {fb} \\\\")
        lines.append(r"\midrule")
        lines.append(f"Totale & {int(dA['gt'].sum())} & {int(dA.tp.sum())} & {int(dA.fn.sum())} & {int(dA.fp.sum())} & "
                     f"{int(dB.tp.sum())} & {int(dB.fn.sum())} & {int(dB.fp.sum())} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nTabelle LaTeX scritte in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Confronto run LLM vs benchmark manuale.")
    ap.add_argument("runs", nargs="+", help="una o due cartelle-run (nome sotto output/ o percorso)")
    ap.add_argument("--gt", default=str(GT_DEFAULT), help="foglio Excel del benchmark manuale")
    ap.add_argument("--tex", default=None, help="con due run: scrive le tabelle LaTeX qui")
    args = ap.parse_args()

    gt = load_gt(Path(args.gt))
    print(f"Benchmark: {len(gt)} righe, {gt.groupby(['doc','cup']).ngroups} coppie (doc,CUP), "
          f"{len(set(zip(gt.doc, gt.cup, gt.imp)))} coppie (doc,CUP,importo) distinte, {gt.doc.nunique()} documenti\n")
    results = [score_run(r, gt) for r in args.runs[:2]]
    print_summary(pd.concat([df for _, df, _ in results]))
    if len(results) == 2 and args.tex:
        (_, dfA, docsA), (_, dfB, docsB) = results
        tex_tables(dfA, dfB, docsA, docsB, Path(args.tex))


if __name__ == "__main__":
    main()
