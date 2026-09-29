"""Cruscotto HTML di tutte le run ECHO, da aprire in locale nel browser.

Uso (nella cartella 5. LLMs):
    python dashboard.py              # aggiorna output/cruscotto.html con tutte le run e lo apre
    python dashboard.py --rifai      # ricalcola anche le run gia' elaborate
    python dashboard.py --no-apri    # solo scrittura

Scorre output/, e per ogni cartella con results.jsonl calcola punteggi e record usando
punteggio.py e confronto.py. Scrive:
    output/cruscotto.html                 la pagina, con elenco run e testi dei prompt
    output/cruscotto_dati/<run>.js        i dati di ogni run, caricati quando la scegli
Le run gia' elaborate si ricalcolano solo se cambiano results.jsonl, il benchmark o
questo script. Le date di emanazione degli atti (per la linea del tempo dei CUP) stanno in
date_decreti.csv, creato alla prima esecuzione e modificabile a mano. Nessun server e nessuna libreria esterna: funziona anche offline.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import json
import math
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path

import confronto as C
import punteggio as P

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
DATI = OUTPUT / "cruscotto_dati"

# Gradino hardware dei modelli (stima a 4 bit, dal baseline del 24/09). I MoE occupano la
# memoria dei parametri TOTALI, non di quelli attivi.
HARDWARE = {
    "google/gemma-4-31b-it":         ("1 GPU", 20),
    "google/gemma-4-26b-a4b-it":     ("1 GPU", 16),
    "z-ai/glm-4.7-flash":            ("1 GPU", 18),
    "qwen/qwen3.6-35b-a3b":          ("1 GPU", 20),
    "deepseek/deepseek-v4-flash":    ("2-4 GPU", 160),
    "moonshotai/kimi-k2.6":          ("8 GPU", 550),
    "deepseek/deepseek-v4-pro-0813": ("Oltre il nodo", 900),
    "moonshotai/kimi-k3":            ("Oltre il nodo", 1500),
    "minimax/minimax-m3":            ("Da accertare", None),
}
ORDINE_GRADINI = ["1 GPU", "2-4 GPU", "8 GPU", "Oltre il nodo", "Da accertare",
                  "Soffitto commerciale"]
# Data di emanazione degli atti del corpus, ricavata il 29/09/2026 dai PDF (non dai modelli).
# affidabilita': "certa" = citata con data in un altro atto o dalla segnatura di protocollo;
# "nome" = solo dal nome del file; "stimata" = ricostruita, con forchetta [min, max].
# Si correggono in date_decreti.csv, che lo script crea alla prima esecuzione e poi legge.
DATE_ATTI = [
    ("2021_DM_mit_464-2021", "2021-11-22", "", "", "certa", "Citato come DM 22 novembre 2021, n. 464 in sette atti del corpus (per esempio DI 97/2022 e DM 191/2023)."),
    ("2022_DM__8_14-01-2022", "2022-01-14", "", "", "certa", "Citato come DM 14 gennaio 2022, n. 8 nel DM 410/2022."),
    ("2022_DI_Mims_Mef_97_20_04_2022", "2022-04-20", "", "", "certa", "Citato come DI 20 aprile 2022, n. 97 in otto atti del corpus."),
    ("2022_DDG_128_del_02-05-22", "2022-05-02", "", "", "nome", "Data nel nome del file; nessun altro atto del corpus lo cita con la data."),
    ("2022_DM_MIMS_342-2022", "2022-10-10", "2022-09-28", "2022-10-22", "stimata", "Cita l'intesa in Conferenza Unificata del 28/09/2022; intestato al Ministero delle infrastrutture e della mobilità sostenibili, quindi precedente al cambio di denominazione di fine ottobre 2022."),
    ("2022_DI_MIT-MEF_390-2022", "2022-11-22", "2022-10-22", "2022-12-21", "stimata", "Intestato al Ministero delle infrastrutture e dei trasporti (denominazione ripresa a fine ottobre 2022) e con numero di registro precedente al DM 409/2022; l'ultimo atto citato è del 05/08/2022."),
    ("2022_DM_MIT_409-2022", "2022-12-22", "2022-12-21", "2022-12-23", "stimata", "Cita l'intesa in Conferenza Unificata del 21/12/2022; numero precedente al DM 410 del 23/12/2022."),
    ("2022_DD_MIT_454-2022", "2022-12-22", "", "", "certa", "Citato come decreto dirigenziale n. 454 del 22 dicembre 2022 nel DD 141/2025."),
    ("2022_DM_410_23-12-2022", "2022-12-23", "", "", "certa", "Citato come DM 23 dicembre 2022, n. 410 nei DM 343/2023 e 330/2024 e nella delibera CIPESS."),
    ("2023_DM_MIT_191-2023", "2023-08-07", "", "", "certa", "Citato come DM 7 agosto 2023, n. 191 nei DM 346/2023 e 330/2024."),
    ("2023_DM_MIT_270-2023", "2023-10-26", "2023-10-19", "2023-10-26", "stimata", "Cita l'intesa in Conferenza Unificata del 19/10/2023; ultima modifica del PDF il 26/10/2023."),
    ("2023_DI_MIT-MEF_286-2023", "2023-11-13", "2023-10-19", "2023-11-13", "stimata", "Numero di registro successivo al DM 270/2023; PDF creato il 24/10/2023 e modificato l'ultima volta il 13/11/2023 (probabile firma del secondo ministro)."),
    ("2023_DM_MIT_346-2023", "2023-12-22", "2023-12-06", "2023-12-22", "stimata", "Cita l'intesa in Conferenza Unificata del 06/12/2023; PDF chiuso il 22/12/2023 come i DM 343 e 345 dello stesso giorno."),
    ("2023_DM_MIT_342-2023", "2023-12-22", "2023-12-20", "2023-12-22", "stimata", "Cita l'intesa in Conferenza Stato-Regioni del 20/12/2023; PDF chiuso il 22/12/2023 come i DM 343 e 345 dello stesso giorno."),
    ("2023_DM_343_22-12-2023", "2023-12-22", "", "", "certa", "Citato come DM 22 dicembre 2023, n. 343 nel DM 330/2024 e nella delibera CIPESS."),
    ("2023_DM_MIT_344-2023", "2023-12-22", "2023-12-20", "2023-12-22", "stimata", "Cita l'intesa in Conferenza Unificata del 20/12/2023; PDF chiuso il 22/12/2023 come i DM 343 e 345 dello stesso giorno."),
    ("2023_DM_MIT_345_-_2023", "2023-12-22", "", "", "certa", "Citato come DM 22 dicembre 2023, n. 345 nel DM 334/2024."),
    ("2023_DI_MIT-MEF_n_362_del_30-12-2023", "2023-12-30", "", "", "certa", "Citato come DI 30 dicembre 2023, n. 362 nei DI 107/2024 e 349/2024."),
    ("2024_DI_MIT_MEF_107-2024", "2024-04-17", "", "", "certa", "Citato come DI 17 aprile 2024, n. 107 nel DI 349/2024."),
    ("2024_delibera_CIPE_del_19_dicembre_2024", "2024-12-19", "", "", "certa", "Data della seduta CIPESS; pubblicata in Gazzetta Ufficiale il 26/03/2025."),
    ("2024_DM_MIT_330_20-12-2024", "2024-12-20", "", "", "nome", "Data nel nome del file, coerente con l'ultima modifica del PDF (20/12/2024)."),
    ("2024_DM_MIT_334-2024", "2024-12-23", "2024-12-18", "2024-12-23", "stimata", "Cita l'intesa in Conferenza Unificata del 18/12/2024; ultima modifica del PDF il 23/12/2024."),
    ("2024_DI_MIT_MEF_349_2024", "2024-12-31", "2024-12-18", "2024-12-31", "stimata", "Cita l'intesa in Conferenza Unificata del 18/12/2024; ultima modifica del PDF il 31/12/2024."),
    ("2025_DM_MIT_131-2025", "2025-04-09", "2025-02-17", "2025-04-18", "stimata", "Ultimo atto citato del 17/02/2025, numero precedente al DD 141 del 18/04/2025; ultima modifica del PDF il 09/04/2025."),
    ("2025_DD_MIT_141-2025", "2025-04-18", "", "", "certa", "Segnatura di protocollo REGISTRO DECRETI R.0000141 del 18-04-2025."),
]
FILE_DATE = ROOT / "date_decreti.csv"
CAMPI_DATE = ["documento", "data", "data_min", "data_max", "affidabilita", "fonte"]


def norm_doc(s: str) -> str:
    """Chiave di confronto fra nomi di file scritti in modi diversi (spazi, punti, underscore)."""
    s = str(s).lower()
    s = s[:-4] if s.endswith(".pdf") else s
    return re.sub(r"[^a-z0-9]", "", s)


def carica_date() -> dict:
    """Legge date_decreti.csv; se manca lo crea dai valori sopra, cosi' si puo' correggere."""
    if not FILE_DATE.exists():
        with FILE_DATE.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(CAMPI_DATE)
            w.writerows(DATE_ATTI)
    date = {}
    with FILE_DATE.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if not (r.get("documento") and r.get("data")):
                continue
            date[norm_doc(r["documento"])] = {
                "data": r["data"].strip(), "min": (r.get("data_min") or "").strip() or None,
                "max": (r.get("data_max") or "").strip() or None,
                "aff": (r.get("affidabilita") or "stimata").strip(), "fonte": (r.get("fonte") or "").strip()}
    return date


MESI = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]


def num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def data_run(run_dir: Path) -> dt.datetime | None:
    """La data sta nel nome della cartella (<AAAAMMGG-hhmmss>__<hash>): e' l'avvio della run.
    Il manifest no: un --resume lo riscrive."""
    try:
        return dt.datetime.strptime(run_dir.name[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def info_prompt(run_dir: Path) -> tuple[str | None, str | None, str | None]:
    """(fingerprint, versione, testo) del prompt della run."""
    mani = {}
    f = run_dir / "run_manifest.json"
    if f.exists():
        try:
            mani = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            mani = {}
    fp = mani.get("prompt_fingerprint") or (run_dir.name.split("__", 1)[1] if "__" in run_dir.name else None)
    testo = None
    if fp and (run_dir / f"{fp}.txt").exists():
        testo = (run_dir / f"{fp}.txt").read_text(encoding="utf-8", errors="replace")
        # le due righe di intestazione cambiano a ogni run: non fanno parte del prompt
        testo = "\n".join(l for l in testo.splitlines() if not l.startswith("# ")).strip("\n")
    return fp, mani.get("prompt_version"), testo


def records_piatti(entries: list[dict]) -> list[dict]:
    """Un dizionario per (record x CUP), con i campi utili al cruscotto."""
    out = []
    for e in entries:
        doc = C.doc_key(e["decree_file"])
        for r in e.get("records") or []:
            cups = r.get("cup")
            cups = [cups] if isinstance(cups, str) else [c for c in (cups or []) if c]
            if not cups:
                for campo in ("cup_provvisorio", "cup_master"):
                    v = r.get(campo)
                    if isinstance(v, str) and v.strip():
                        cups = [v.strip()]
                        break
            imp = num(r.get("importo_eur"))
            if imp is None or not cups:
                continue
            cap_pg = C.norm_cap_pg(r.get("capitolo"), r.get("piano_gestionale"))
            for c in cups:
                out.append({
                    "m": e["model_slug"], "d": doc, "cup": str(c).strip().upper(),
                    "imp": round(imp / len(cups), 2), "nCup": len(cups),
                    "cap": cap_pg.split("/")[0] if cap_pg else None, "pg": cap_pg,
                    "liv": r.get("livello_importo") or "totale",
                    "stato": r.get("stato_importo") or "altro",
                    "rif": (r.get("riferimento_fonte") or "")[:160],
                    "txt": (r.get("passaggio_testuale") or "")[:200],
                })
    return out


def elabora(run_dir: Path, gt_path: Path, gt) -> dict:
    _, entries = C.load_run(str(run_dir))
    punt = P.valuta(str(run_dir), gt_path, P.QUARANTENA_DEFAULT)
    eseguiti = sorted({C.doc_key(e["decree_file"]) for e in entries})
    gt_eseguiti = gt[gt["doc"].isin(eseguiti)]

    per_modello = defaultdict(list)
    for e in entries:
        per_modello[e["model_slug"]].append(e)

    modelli, matrice = [], {}
    for slug, ents in per_modello.items():
        cella = punt["celle"].get(f"{slug}::{ents[0].get('thinking')}", {})
        s, _ = C.score_model(ents, gt_eseguiti)
        kind = ents[0].get("model_kind") or ("open" if slug in HARDWARE else "commerciale")
        gradino, vram = HARDWARE.get(slug, ("Soffitto commerciale" if kind != "open" else "Da accertare", None))
        stati = Counter(e.get("status") for e in ents)
        modelli.append({
            "id": slug, "nome": slug.split("/")[-1], "kind": kind, "gradino": gradino, "vram": vram,
            "solved": cella.get("solved", 0), "su": cella.get("su", 0),
            "f1": round(s["pairtot_F1"], 3), "cupF1": round(s["cup_F1"], 3),
            "costo": round(s["cost_usd"] or 0, 4), "lat": round(s["latency_mean_s"] or 0, 1),
            "tok": int(s["completion_tokens"] or 0),
            "celle": len(ents), "celleOk": stati.get("ok", 0) + stati.get("ok_repaired", 0),
            "inventati": cella.get("guardia_violata", []),
        })
        matrice[slug] = {
            r["doc"]: {"ok": r["solved"], "gt": r["gt"], "pred": r["pred"], "fn": r["fn"], "fp": r["fp"],
                       "st": r["status"],
                       "fnR": [[x[0], x[1], x[2]] for x in r["fn_righe"][:15]],
                       "fpR": [[x[0], x[1], x[2]] for x in r["fp_righe"][:15]]}
            for r in cella.get("dettaglio", [])
        }

    con_bench = set(gt["doc"])
    return {
        "meta": {"run": run_dir.name, "prompt": info_prompt(run_dir)[0], "nDoc": len(eseguiti),
                 "nModelli": len(modelli), "costo": round(sum(m["costo"] for m in modelli), 3),
                 "quarantena": punt["quarantena"]},
        "gradini": ORDINE_GRADINI,
        "modelli": modelli,
        "docBench": punt["eleggibili"],
        "docSenzaCup": [d for d in eseguiti if d not in con_bench],
        "matrice": matrice,
        "rec": records_piatti(entries),
        "gt": [{"d": r.doc, "cup": r.cup, "imp": r.imp,
                "pg": r.cap_pg if isinstance(r.cap_pg, str) else None,
                "stato": r.tipologia if isinstance(r.tipologia, str) else None}
               for r in gt[gt["doc"].isin(eseguiti)].itertuples()],
    }


def sommario(run_dir: Path, dati: dict) -> dict:
    d = data_run(run_dir)
    fp, versione, _ = info_prompt(run_dir)
    M = dati["modelli"]
    ordina = lambda xs: sorted(xs, key=lambda m: (-m["solved"], -m["f1"], m["costo"]))
    aperti = ordina([m for m in M if m["kind"] == "open"])
    tutti = ordina(M)
    breve = lambda m: {"nome": m["nome"], "solved": m["solved"], "su": m["su"]} if m else None
    return {
        "run": run_dir.name, "fp": fp, "versione": versione,
        "ordine": d.isoformat() if d else run_dir.name,
        "dataBreve": f"{d.day} {MESI[d.month-1]} {d:%H:%M}" if d else run_dir.name,
        "dataEstesa": f"{d:%d/%m/%Y} alle {d:%H:%M}" if d else run_dir.name,
        "nDoc": dati["meta"]["nDoc"], "nModelli": len(M), "costo": dati["meta"]["costo"],
        "miglioreAperto": breve(aperti[0] if aperti else None),
        "miglioreTot": breve(tutti[0] if tutti else None),
        "modelli": [{"nome": m["nome"], "kind": m["kind"], "solved": m["solved"], "su": m["su"], "f1": m["f1"]}
                    for m in M],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=C.GT_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUTPUT, help="cartella delle run (default: output)")
    ap.add_argument("--rifai", action="store_true", help="ricalcola tutte le run")
    ap.add_argument("--no-apri", action="store_true")
    a = ap.parse_args()

    out_dir, dati_dir = a.output, a.output / "cruscotto_dati"
    dati_dir.mkdir(parents=True, exist_ok=True)
    gt = C.load_gt(a.gt)
    gt = gt[gt["doc"].notna() & (gt["doc"].astype(str) != "nan")]
    riferimento = max(a.gt.stat().st_mtime, Path(__file__).stat().st_mtime)

    runs = sorted(d for d in out_dir.iterdir() if d.is_dir() and (d / "results.jsonl").exists())
    indice, prompt = [], {}
    for run_dir in runs:
        js, som = dati_dir / f"{run_dir.name}.js", dati_dir / f"{run_dir.name}.sommario.json"
        aggiornato = (js.exists() and som.exists() and not a.rifai
                      and js.stat().st_mtime >= max(riferimento, (run_dir / "results.jsonl").stat().st_mtime))
        if aggiornato:
            s = json.loads(som.read_text(encoding="utf-8"))
        else:
            try:
                dati = elabora(run_dir, a.gt, gt)
            except Exception as e:  # una run vecchia o incompleta non deve fermare le altre
                print(f"  salto {run_dir.name}: {type(e).__name__}: {e}")
                continue
            blob = json.dumps(dati, ensure_ascii=False, default=str).replace("</", "<\\/")
            js.write_text(f"window.ECHO_RUN({blob});\n", encoding="utf-8")
            s = sommario(run_dir, dati)
            som.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
            print(f"  elaborata {run_dir.name}")
        indice.append(s)
        fp, versione, testo = info_prompt(run_dir)
        if fp and testo and fp not in prompt:
            prompt[fp] = {"versione": versione, "testo": testo}
        if fp in prompt:
            prompt[fp]["versione"] = prompt[fp]["versione"] or versione

    indice.sort(key=lambda s: s["ordine"])
    for s in indice:  # data della prima run di ogni prompt
        if s["fp"] in prompt and "prima" not in prompt[s["fp"]]:
            prompt[s["fp"]]["prima"] = s["dataEstesa"]
    blob = json.dumps({"runs": indice, "prompt": prompt, "date": carica_date(),
                       "generato": dt.datetime.now().strftime("%d/%m/%Y alle %H:%M")},
                      ensure_ascii=False).replace("</", "<\\/")
    pagina = out_dir / "cruscotto.html"
    pagina.write_text(TEMPLATE.replace("/*__INDICE__*/null", blob), encoding="utf-8")
    print(f"Cruscotto con {len(indice)} run scritto in {pagina}")
    if not a.no_apri:
        webbrowser.open(pagina.resolve().as_uri())


TEMPLATE = r'''<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ECHO, cruscotto delle run</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --notte:#1b1526; --fondo:#241b2f; --pannello:#2a2139; --rialzo:#34294f; --linea:#463a63;
  --rosa:#ff7edb; --magenta:#f92aad; --ciano:#36f9f6; --ciano2:#03edf9; --verde:#72f1b8;
  --giallo:#fede5d; --arancio:#f97e72; --lavanda:#848bbd; --testo:#f4eee4; --tenue:#b6b1c9;
  --titolo:"Chakra Petch","Segoe UI",system-ui,sans-serif;
  --corpo:"Barlow","Segoe UI",system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box}
html{background:var(--notte)}
body{margin:0;color:var(--testo);font:16px/1.5 var(--corpo);min-height:100vh;
  background:radial-gradient(ellipse 90% 60% at 50% 0,#34294f 0,transparent 70%),var(--fondo);
  font-variant-numeric:tabular-nums}
.wrap{max-width:1320px;margin:0 auto;padding:0 28px 80px}
a{color:var(--ciano)}
:focus-visible{outline:2px solid var(--ciano);outline-offset:2px}

/* testata */
header{padding:48px 0 28px;position:relative}
header::after{content:"";display:block;height:3px;margin-top:28px;
  background:linear-gradient(90deg,var(--magenta),var(--rosa) 35%,var(--giallo) 60%,var(--ciano2));
  box-shadow:0 0 18px #f92aad66}
.sigla{font:700 15px/1 var(--titolo);letter-spacing:.08em;color:var(--rosa);margin:0 0 14px}
h1{font:700 clamp(28px,4vw,46px)/1.12 var(--titolo);margin:0;max-width:26ch;
  text-shadow:0 0 2px #100c0f,0 0 10px #fc28a855,0 0 26px #fc28a833}
h1 b{color:var(--ciano);font-weight:700;text-shadow:0 0 3px #001716,0 0 8px #03edf975,0 0 18px #03edf955}
.meta{color:var(--tenue);margin:14px 0 0;font-size:15px}
.gradini{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-top:30px}
.gradino{padding:12px 14px;border-left:3px solid var(--linea);background:#2a213999}
.gradino h3{margin:0;font:600 14px/1.2 var(--titolo);color:var(--tenue)}
.gradino .v{font:700 26px/1.1 var(--titolo);margin-top:6px}
.gradino .n{font-size:14px;color:var(--tenue)}
.gradino.top{border-left-color:var(--ciano)}
.gradino.top .v{color:var(--ciano)}
.gradino.soffitto .v{color:var(--giallo)}

/* sezioni */
section{margin-top:56px}
h2{font:600 24px/1.2 var(--titolo);margin:0 0 6px}
.nota{color:var(--tenue);margin:0 0 18px;max-width:78ch;font-size:15px}
.due{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,.8fr);gap:36px;align-items:start}
@media (max-width:1100px){.due{grid-template-columns:1fr}}
#scatter{max-width:620px;display:block}
.scorri{overflow-x:auto}

/* tabelle */
table{border-collapse:collapse;width:100%;font-size:15px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #46396355;white-space:nowrap}
th{font:600 13px/1.2 var(--titolo);color:var(--tenue);position:sticky;top:0;background:var(--pannello)}
th button{all:unset;cursor:pointer}
th button[aria-sort]::after{content:" ▾";color:var(--rosa)}
th button[aria-sort=ascending]::after{content:" ▴"}
td.n,th.n{text-align:right}
tbody tr:hover{background:#34294f88}
.barra{display:flex;align-items:center;gap:8px}
.barra i{display:block;height:8px;background:linear-gradient(90deg,var(--magenta),var(--rosa));border-radius:4px;min-width:2px}
.barra.best i{background:linear-gradient(90deg,var(--ciano2),var(--verde));box-shadow:0 0 8px #03edf966}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:13px;background:var(--rialzo);color:var(--tenue)}
.tag.aperto{color:var(--verde)} .tag.chiuso{color:var(--giallo)}
.avviso{color:var(--arancio)}
.ok{color:var(--verde)}

/* grafico efficienza */
svg text{font-family:var(--corpo);fill:var(--tenue);font-size:12px}
.asse line,.asse path{stroke:#463a63}
.punto{cursor:default}
.punto circle{transition:r .15s}
.punto:hover circle{stroke:#fff;stroke-width:2}

/* matrice */
.matrice{display:grid;gap:3px;font-size:13px;min-width:760px}
.matrice .intest{writing-mode:vertical-rl;transform:rotate(180deg);color:var(--tenue);font-size:12px;
  height:140px;padding:4px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;justify-self:center}
.matrice .mod{padding:6px 10px 6px 0;white-space:nowrap;align-self:center}
.cella{height:34px;display:flex;align-items:center;justify-content:center;border-radius:4px;
  background:#34294f;color:var(--tenue);font-size:12px;cursor:default}
.cella.ris{background:#72f1b8;color:#1b1526;font-weight:600;box-shadow:0 0 10px #72f1b844}
.cella.no{background:#f92aad26;border:1px solid #f92aad88;color:var(--rosa)}
.cella.guasta{background:repeating-linear-gradient(45deg,#f97e7233 0 5px,transparent 5px 10px);
  border:1px solid var(--arancio);color:var(--arancio)}
.cella.vuota{background:transparent;border:1px dashed #46396399}
.cella.inv{background:#fede5d22;border:1px solid var(--giallo);color:var(--giallo)}
.cella.pulita{background:#72f1b81f;color:var(--verde)}
.legenda{display:flex;flex-wrap:wrap;gap:16px;margin-top:12px;font-size:14px;color:var(--tenue)}
.legenda span{display:inline-flex;align-items:center;gap:6px}
.legenda i{width:14px;height:14px;border-radius:3px;display:inline-block}

/* flusso */
.comandi{display:flex;flex-wrap:wrap;gap:12px 18px;align-items:end;margin-bottom:14px}
.comandi label{display:flex;flex-direction:column;gap:4px;font-size:13px;color:var(--tenue)}
.comandi label.spunta{flex-direction:row;align-items:center;gap:8px;font-size:14px}
select,input[type=search]{font:15px var(--corpo);color:var(--testo);background:var(--notte);
  border:1px solid var(--linea);border-radius:6px;padding:7px 10px;min-width:200px}
input[type=checkbox]{accent-color:var(--rosa);width:16px;height:16px}
.palco{background:radial-gradient(ellipse at 50% 120%,#f92aad22,transparent 60%),#1b1526cc;
  border:1px solid var(--linea);border-radius:10px;padding:18px 10px;overflow-x:auto}
.sintesi{display:flex;flex-wrap:wrap;gap:8px 26px;margin:0 0 12px;color:var(--tenue);font-size:15px}
.sintesi b{color:var(--testo);font-weight:600}
#flusso{display:block;min-width:900px}
#flusso .arco{fill:none;stroke-linecap:butt;transition:opacity .15s}
#flusso .scia{fill:none;stroke:#fff;stroke-opacity:.55;stroke-dasharray:2 16;animation:scorre 1.6s linear infinite;pointer-events:none}
@keyframes scorre{to{stroke-dashoffset:-18}}
@media (prefers-reduced-motion:reduce){#flusso .scia{animation:none;display:none}}
#flusso .nodo rect{rx:5;stroke-width:1.5}
#flusso .nodo text{fill:var(--testo);font-size:12.5px}
#flusso .nodo .sub{fill:var(--tenue);font-size:11px}
#flusso .nodo{cursor:pointer}
#flusso .grp,#flusso .nodo{transition:opacity .15s}
#flusso.fuoco .grp{opacity:.07}
#flusso.fuoco .nodo{opacity:.25}
#flusso.fuoco .caldo{opacity:1}
#flusso .colonna{font:600 13px var(--titolo);fill:var(--rosa)}
.vuoto{padding:40px;text-align:center;color:var(--tenue)}

/* scheda CUP */
.scheda{margin-top:18px;border:1px solid var(--linea);border-left:3px solid var(--ciano);
  border-radius:8px;padding:16px 18px;background:#2a2139cc}
.scheda h3{margin:0 0 4px;font:700 20px var(--titolo);color:var(--ciano)}
.scheda .scorri{max-height:440px;overflow:auto}
.scheda .chiudi{float:right;background:none;border:1px solid var(--linea);color:var(--tenue);
  border-radius:6px;padding:4px 10px;cursor:pointer;font:14px var(--corpo)}

/* schede tabelle */
.linguette{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;align-items:end}
.linguette button{font:600 15px var(--titolo);color:var(--tenue);background:none;border:none;
  border-bottom:2px solid transparent;padding:6px 12px;cursor:pointer}
.linguette button[aria-selected=true]{color:var(--testo);border-bottom-color:var(--rosa)}
.linguette input{margin-left:auto}
.tabellona{max-height:560px;overflow:auto;border:1px solid var(--linea);border-radius:8px}
tr.cliccabile{cursor:pointer}
code{font-family:var(--corpo);font-weight:600;letter-spacing:.02em;color:var(--testo)}

/* tooltip */
#tip{position:fixed;z-index:10;pointer-events:none;max-width:380px;background:#1b1526f2;
  border:1px solid var(--rosa);border-radius:8px;padding:10px 12px;font-size:14px;line-height:1.4;
  box-shadow:0 0 18px #f92aad44;opacity:0;transition:opacity .1s}
#tip b{color:var(--ciano)}
#tip .t{color:var(--tenue);font-size:13px;margin-top:4px}
footer{margin-top:60px;color:var(--tenue);font-size:14px}

/* linea del tempo */
.scelte{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px}
.chip{display:inline-flex;align-items:center;gap:6px;padding:4px 6px 4px 10px;border-radius:14px;
  background:var(--rialzo);border:1px solid var(--ciano2);font:600 13px var(--corpo)}
.chip button{all:unset;cursor:pointer;color:var(--tenue);padding:0 4px;font-size:15px;line-height:1}
.chip button:hover{color:var(--rosa)}
#tempo{display:block;min-width:900px}
#tempo .corsia{fill:#34294f55}
#tempo .vita{stroke:#36f9f6;stroke-opacity:.35;stroke-width:2}
#tempo .evento{cursor:pointer}
#tempo .evento:hover circle,#tempo .evento:focus circle{stroke:#fff;stroke-width:2}
#tempo .forchetta{stroke-width:2;stroke-opacity:.6}
#tempo .anno{fill:var(--rosa);font:600 12px var(--titolo)}
#tempo .atto{stroke:#848bbd;stroke-opacity:.5}
#tempo .etic{font-size:11px;fill:var(--tenue)}
details.date{margin-top:14px;color:var(--tenue)}
details.date summary{cursor:pointer;font:600 14px var(--titolo);color:var(--testo)}
.aff{font-size:12px;padding:1px 7px;border-radius:9px}
.aff.certa{background:#72f1b822;color:var(--verde)}
.aff.nome{background:#36f9f622;color:var(--ciano)}
.aff.stimata{background:#fede5d22;color:var(--giallo)}

/* scelta della run e prompt */
.barra-run{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:0 0 26px}
.barra-run label{display:flex;flex-direction:column;gap:4px;font-size:13px;color:var(--tenue)}
.barra-run select{min-width:min(460px,80vw)}
.bottone{font:600 14px var(--titolo);color:var(--testo);background:var(--rialzo);border:1px solid var(--rosa);
  border-radius:6px;padding:8px 14px;cursor:pointer;box-shadow:0 0 10px #f92aad33}
.bottone:hover{background:#463a63}
tr.corrente td{background:#03edf912}
tr.corrente td:first-child{box-shadow:inset 3px 0 0 var(--ciano)}
.linkp{all:unset;cursor:pointer;color:var(--ciano);text-decoration:underline dotted}
#evoluzione{display:block;width:100%;height:auto;margin-bottom:14px}
dialog{width:min(1100px,94vw);max-height:88vh;padding:0;border:1px solid var(--rosa);border-radius:10px;
  background:var(--pannello);color:var(--testo);box-shadow:0 0 40px #f92aad44}
dialog::backdrop{background:#1b1526cc}
.dlg-testa{display:flex;flex-wrap:wrap;gap:12px 18px;align-items:end;padding:16px 18px;border-bottom:1px solid var(--linea);
  position:sticky;top:0;background:var(--pannello)}
.dlg-testa h3{margin:0;font:700 19px var(--titolo);margin-right:auto}
.dlg-testa label{display:flex;flex-direction:column;gap:4px;font-size:13px;color:var(--tenue)}
.dlg-corpo{overflow:auto;max-height:calc(88vh - 90px)}
pre#testoPrompt{margin:0;padding:16px 18px;font:13px/1.55 "Cascadia Mono",Consolas,"Courier New",monospace;white-space:pre-wrap}
pre#testoPrompt .add{display:block;background:#72f1b81c;color:var(--verde)}
pre#testoPrompt .del{display:block;background:#f92aad1c;color:var(--rosa)}
pre#testoPrompt .salto{display:block;color:var(--lavanda);padding:4px 0}
</style>
</head>
<body>
<div id="tip" role="tooltip"></div>
<dialog id="dlgPrompt" aria-labelledby="dlgTitolo">
  <div class="dlg-testa">
    <h3 id="dlgTitolo"></h3>
    <label>Confronta con<select id="selDiff"></select></label>
    <button class="bottone" id="chiudiPrompt" type="button">Chiudi</button>
  </div>
  <div class="dlg-corpo"><pre id="testoPrompt"></pre></div>
</dialog>
<div class="wrap">
  <header>
    <p class="sigla">ECHO · estrazione dai decreti</p>
    <div class="barra-run">
      <label>Run da visualizzare<select id="selRun"></select></label>
      <button class="bottone" id="apriPrompt" type="button">Vedi il prompt</button>
    </div>
    <h1 id="titolo"></h1>
    <p class="meta" id="meta"></p>
    <div class="gradini" id="gradini"></div>
  </header>

  <section id="storico">
    <h2>Storico delle run</h2>
    <p class="nota">Ogni punto è un modello in una run, come quota di documenti esatti sul totale valutato in quella run. Le run hanno denominatori diversi (per esempio 5 documenti pilota o 16 con benchmark): confronta le quote, non i numeri assoluti. La linea segue il miglior modello aperto. Clicca una riga per aprire la run.</p>
    <svg id="evoluzione" viewBox="0 0 1100 270" role="img" aria-label="Andamento delle run"></svg>
    <div class="tabellona" style="max-height:360px"><table id="tabRun"></table></div>
  </section>

  <section>
      <div>
        <h2>Classifica dei modelli</h2>
        <p class="nota">Un documento conta come esatto quando l'insieme delle coppie CUP e importo coincide con quello di Alessandro, senza righe mancanti né in più (tolleranza 1 €). Clicca un'intestazione per ordinare.</p>
        <div class="scorri"><table id="classifica"></table></div>
      </div>
  </section>

  <section>
    <div class="due">
      <div>
        <h2>Efficienza</h2>
        <p class="nota">Costo della run sull'asse orizzontale (scala logaritmica), documenti esatti su quello verticale. Il cerchio è proporzionale alla memoria necessaria per installare il modello: in alto a sinistra e piccolo è il posto migliore.</p>
        <svg id="scatter" viewBox="0 0 560 400" role="img" aria-label="Costo contro documenti esatti"></svg>
      </div>
      <div>
        <h2>Dove sbagliano</h2>
        <p class="nota">Righe mancanti e righe in più sommate su tutti i documenti con benchmark. Molte righe in più indicano che il modello registra somme solo richiamate; molte mancanti, che perde righe di tabella.</p>
        <div id="errori"></div>
      </div>
    </div>
  </section>

  <section>
    <h2>Documento per documento</h2>
    <p class="nota">Verde: identico al benchmark. Rosa: numero di righe mancanti (−) e in più (+). Passa sopra una cella per vedere quali. A destra, gli atti senza CUP nel benchmark: qualunque CUP trovato lì è inventato.</p>
    <div class="scorri"><div class="matrice" id="matrice"></div></div>
    <div class="legenda">
      <span><i style="background:#72f1b8"></i>esatto</span>
      <span><i style="background:#f92aad26;border:1px solid #f92aad88"></i>diverso dal benchmark</span>
      <span><i style="background:repeating-linear-gradient(45deg,#f97e7255 0 3px,transparent 3px 6px);border:1px solid #f97e72"></i>cella non riuscita (errore o risposta vuota)</span>
      <span><i style="background:#fede5d22;border:1px solid #fede5d"></i>CUP inventati su atto senza CUP</span>
    </div>
  </section>

  <section>
    <h2>Dal piano gestionale al CUP</h2>
    <p class="nota">Ogni freccia è un evento contabile estratto: da quale capitolo e piano gestionale partono le risorse, verso quale progetto. Lo spessore cresce con l'importo, il colore dice lo stato. Clicca un CUP per la sua scheda.</p>
    <div class="comandi">
      <label>Fonte<select id="fFonte"></select></label>
      <label>Documento<select id="fDoc"></select></label>
      <label>Stato<select id="fStato"></select></label>
      <label class="spunta"><input type="checkbox" id="fNoPg">includi i record senza PG</label>
    </div>
    <div class="palco">
      <div class="sintesi" id="sintesi"></div>
      <svg id="flusso" role="img" aria-label="Flussi dal piano gestionale al CUP"></svg>
    </div>
    <div class="legenda" id="legStati"></div>
    <div id="scheda"></div>
  </section>

  <section id="sezTempo">
    <h2>Linea del tempo dei CUP</h2>
    <p class="nota">Scegli uno o più CUP per vedere, atto per atto, quando le risorse sono state assegnate, impegnate, rimodulate o revocate. Ogni cerchio è un atto: la data è quella di emanazione, la dimensione cresce con l'importo, il colore dice lo stato. Un cerchio vuoto con la forchetta indica una data stimata.</p>
    <div class="comandi">
      <label>Fonte<select id="tFonte"></select></label>
      <label>Aggiungi un CUP<input type="search" id="tCup" list="tCupElenco" placeholder="Scrivi o scegli un CUP" aria-label="Aggiungi un CUP"></label>
      <datalist id="tCupElenco"></datalist>
      <button class="bottone" id="tAggiungi" type="button">Aggiungi</button>
      <button class="bottone" id="tPiuAttivi" type="button">I 6 CUP con più atti</button>
      <button class="bottone" id="tSvuota" type="button">Svuota</button>
    </div>
    <div class="scelte" id="tScelti"></div>
    <div class="palco"><svg id="tempo" role="img" aria-label="Linea del tempo dei CUP scelti"></svg></div>
    <div class="legenda" id="tLegenda"></div>
    <h3 style="font:600 18px var(--titolo);margin:26px 0 8px">Cronologia</h3>
    <div class="tabellona" style="max-height:420px"><table id="tCrono"></table></div>
    <details class="date"><summary>Date degli atti e come sono state ricavate</summary><div class="scorri" style="margin-top:10px"><table id="tDate"></table></div></details>
  </section>

  <section>
    <h2>CUP e piani gestionali trovati</h2>
    <p class="nota">Importi e PG vengono dalla lettura condivisa dalla maggioranza dei modelli aperti; «Modelli» dice quanti l'hanno trovato in qualunque forma. Clicca una riga per la scheda del CUP.</p>
    <div class="linguette" role="tablist">
      <button role="tab" aria-selected="true" data-t="cup">CUP</button>
      <button role="tab" aria-selected="false" data-t="pg">Piani gestionali</button>
      <input type="search" id="cerca" placeholder="Cerca CUP, PG o documento" aria-label="Cerca">
    </div>
    <div class="tabellona"><table id="elenco"></table></div>
  </section>

  <footer id="piede"></footer>
</div>

<script>
const INDICE = /*__INDICE__*/null;
let D = null;
const CACHE = {};
let M, aperti, ordinati, migliorAperto, soglia, consenso, benchCup, cupRighe, pgRighe;

/* ---------- utilità ---------- */
const $ = s => document.querySelector(s);
const eur0 = new Intl.NumberFormat("it-IT",{style:"currency",currency:"EUR",maximumFractionDigits:0});
const eur = v => eur0.format(v);
function eurBreve(v){
  const a = Math.abs(v);
  if (a >= 1e9) return (v/1e9).toLocaleString("it-IT",{maximumFractionDigits:2})+" mld €";
  if (a >= 1e6) return (v/1e6).toLocaleString("it-IT",{maximumFractionDigits:1})+" mln €";
  if (a >= 1e3) return (v/1e3).toLocaleString("it-IT",{maximumFractionDigits:0})+" mila €";
  return eur(v);
}
const usd = v => "$" + v.toLocaleString("it-IT",{minimumFractionDigits:2,maximumFractionDigits:2});
const pct = v => (v*100).toLocaleString("it-IT",{maximumFractionDigits:1}) + "%";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const breve = d => d.replace(/^\d{4}\s+/, "").replace(/\s+del\s+/, " ");
const SVG = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent){
  const n = document.createElementNS(SVG, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
const tip = $("#tip");
function mostra(html, ev){
  tip.innerHTML = html; tip.style.opacity = 1; sposta(ev);
}
function sposta(ev){
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + 16, y = ev.clientY + 16;
  if (x + w > innerWidth - 8) x = ev.clientX - w - 16;
  if (y + h > innerHeight - 8) y = ev.clientY - h - 16;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const nascondi = () => tip.style.opacity = 0;
function conTip(node, fn){
  node.addEventListener("mouseenter", e => mostra(fn(), e));
  node.addEventListener("mousemove", sposta);
  node.addEventListener("mouseleave", nascondi);
}

const COL_STATO = {impegnato:"#36f9f6", assegnato:"#ff7edb", rimodulato:"#fede5d",
  revocato:"#f97e72", ammesso:"#72f1b8", altro:"#848bbd"};
const colStato = s => COL_STATO[s] || COL_STATO.altro;

/* ---------- preparazione ---------- */
const nome = id => (M.find(m => m.id === id) || {nome:id}).nome;
const chiave = r => [r.d, r.cup, r.pg || "", Math.round(r.imp)].join("|");
function prepara(){
  M = D.modelli;
  aperti = M.filter(m => m.kind === "open");
  ordinati = [...M].sort((a,b) => b.solved - a.solved || b.f1 - a.f1 || a.costo - b.costo);
  migliorAperto = ordinati.find(m => m.kind === "open");
  // lettura condivisa: record su cui concorda la maggioranza dei modelli aperti
  soglia = Math.max(1, Math.ceil(aperti.length / 2));
  const perChiave = new Map();
  for (const r of D.rec){
    if (!aperti.some(m => m.id === r.m)) continue;
    const k = chiave(r);
    if (!perChiave.has(k)) perChiave.set(k, {r, mod:new Set()});
    perChiave.get(k).mod.add(r.m);
  }
  consenso = [...perChiave.values()].filter(x => x.mod.size >= soglia)
    .map(x => ({...x.r, m:"consenso", nMod:x.mod.size}));
  benchCup = new Set(D.gt.map(g => g.cup));
  const mc = new Map();
  for (const r of D.rec){
    if (!mc.has(r.cup)) mc.set(r.cup, {cup:r.cup, docs:new Set(), mod:new Set(), pg:new Set(), imp:0});
    const x = mc.get(r.cup); x.docs.add(r.d); x.mod.add(r.m);
  }
  for (const r of consenso){ const x = mc.get(r.cup); if (!x) continue; x.imp += Math.abs(r.imp); if (r.pg) x.pg.add(r.pg); }
  cupRighe = [...mc.values()].map(x => ({...x, bench:benchCup.has(x.cup), nMod:x.mod.size}));
  const mp = new Map();
  for (const r of consenso){
    if (!r.pg) continue;
    if (!mp.has(r.pg)) mp.set(r.pg, {pg:r.pg, cap:r.cap, cups:new Set(), docs:new Set(), imp:0, stati:{}});
    const x = mp.get(r.pg); x.cups.add(r.cup); x.docs.add(r.d); x.imp += Math.abs(r.imp);
    x.stati[r.stato] = (x.stati[r.stato] || 0) + 1;
  }
  const modPerPg = new Map();
  for (const r of D.rec){ if (!r.pg) continue; (modPerPg.get(r.pg) || modPerPg.set(r.pg, new Set()).get(r.pg)).add(r.m); }
  for (const [pg] of modPerPg) if (!mp.has(pg)) mp.set(pg, {pg, cap:pg.split("/")[0], cups:new Set(), docs:new Set(), imp:0, stati:{}});
  pgRighe = [...mp.values()].map(x => ({...x, nMod:(modPerPg.get(x.pg) || new Set()).size}));
}

/* ---------- testata ---------- */
function testata(){
  const b = migliorAperto;
  $("#titolo").innerHTML = b
    ? `Il miglior modello aperto, <b>${esc(b.nome)}</b>, riproduce il benchmark su ${b.solved} documenti su ${b.su}.`
    : "Nessun modello aperto in questa run.";
  const m = D.meta, info = INDICE.runs.find(r => r.run === m.run) || {};
  $("#meta").textContent = `Run del ${info.dataEstesa || m.run}, prompt ${info.versione || "senza versione"} (${m.prompt || "n.d."}), ${m.nDoc} documenti, ${m.nModelli} modelli, spesa ${usd(m.costo)}` +
    (m.quarantena.length ? `. In quarantena: ${m.quarantena.join(", ")}.` : ".");
  const box = $("#gradini"); box.innerHTML = "";
  for (const g of D.gradini){
    const qui = M.filter(x => x.gradino === g);
    if (!qui.length) continue;
    const top = [...qui].sort((a,b) => b.solved - a.solved || b.f1 - a.f1)[0];
    const d = document.createElement("div");
    d.className = "gradino" + (top === b ? " top" : "") + (top.kind !== "open" ? " soffitto" : "");
    d.innerHTML = `<h3>${esc(g)}</h3><div class="v">${top.solved}/${top.su}</div>
      <div class="n">${esc(top.nome)}${top.vram ? `, circa ${top.vram} GB` : ""}</div>`;
    box.appendChild(d);
  }
}

/* ---------- classifica ---------- */
const COLONNE = [
  {k:"nome", t:"Modello", f:m => `${esc(m.nome)} <span class="tag ${m.kind==="open"?"aperto":"chiuso"}">${m.kind==="open"?"aperto":"commerciale"}</span>`},
  {k:"gradino", t:"Gradino", f:m => esc(m.gradino)},
  {k:"solved", t:"Documenti esatti", n:1, f:m => {
    const w = m.su ? 110 * m.solved / m.su : 0;
    return `<span class="barra ${m===migliorAperto?"best":""}"><i style="width:${w}px"></i>${m.solved}/${m.su}</span>`;}},
  {k:"f1", t:"F1 coppie", n:1, f:m => m.f1.toLocaleString("it-IT",{minimumFractionDigits:3})},
  {k:"cupF1", t:"F1 CUP", n:1, f:m => m.cupF1.toLocaleString("it-IT",{minimumFractionDigits:3})},
  {k:"costo", t:"Costo", n:1, f:m => usd(m.costo)},
  {k:"lat", t:"Latenza media", n:1, f:m => m.lat.toLocaleString("it-IT") + " s"},
  {k:"tok", t:"Token in uscita", n:1, f:m => m.tok.toLocaleString("it-IT")},
  {k:"guasti", t:"Celle non riuscite", n:1, f:m => {
    const g = m.celle - m.celleOk; return g ? `<span class="avviso">${g}</span>` : `<span class="ok">0</span>`;}},
  {k:"inv", t:"Atti con CUP inventati", n:1, f:m => m.inventati.length ? `<span class="avviso">${m.inventati.length}</span>` : `<span class="ok">0</span>`},
];
let ordine = {k:"solved", dir:-1};
function valore(m, k){ return k==="guasti" ? m.celle - m.celleOk : k==="inv" ? m.inventati.length : m[k]; }
function classifica(){
  const righe = [...M].sort((a,b) => {
    const x = valore(a, ordine.k), y = valore(b, ordine.k);
    const c = typeof x === "string" ? x.localeCompare(y) : x - y;
    return c * ordine.dir || b.f1 - a.f1;
  });
  const t = $("#classifica");
  t.innerHTML = `<thead><tr>${COLONNE.map(c => `<th class="${c.n?"n":""}"><button data-k="${c.k}" ${ordine.k===c.k?`aria-sort="${ordine.dir>0?"ascending":"descending"}"`:""}>${c.t}</button></th>`).join("")}</tr></thead>
    <tbody>${righe.map(m => `<tr>${COLONNE.map(c => `<td class="${c.n?"n":""}">${c.f(m)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  t.querySelectorAll("th button").forEach(b => b.onclick = () => {
    const k = b.dataset.k;
    ordine = {k, dir: ordine.k === k ? -ordine.dir : (k==="nome"||k==="gradino"||k==="costo"||k==="lat"||k==="tok"||k==="guasti"||k==="inv" ? 1 : -1)};
    classifica();
  });
}

/* ---------- efficienza ---------- */
function scatter(){
  const s = $("#scatter"); s.innerHTML = ""; const W = 560, H = 400, L = 54, R = 20, T = 16, B = 46;
  const costi = M.map(m => Math.max(m.costo, 1e-4));
  const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...costi))));
  const hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...costi))));
  const su = Math.max(...M.map(m => m.su), 1);
  const X = v => L + (Math.log10(Math.max(v,1e-4)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (W - L - R);
  const Y = v => T + (1 - v / su) * (H - T - B);
  const g = el("g", {class:"asse"}, s);
  for (let e = Math.log10(lo); e <= Math.log10(hi) + 1e-9; e++){
    const x = X(Math.pow(10, e));
    el("line", {x1:x, x2:x, y1:T, y2:H-B, "stroke-dasharray":"2 4"}, g);
    const v = Math.pow(10, e);
    el("text", {x, y:H-B+18, "text-anchor":"middle"}, g).textContent = "$" + v.toLocaleString("it-IT", {maximumFractionDigits: Math.max(0, -e)});
  }
  for (let v = 0; v <= su; v += Math.max(1, Math.round(su/4))){
    const y = Y(v);
    el("line", {x1:L, x2:W-R, y1:y, y2:y, "stroke-dasharray":"2 4"}, g);
    el("text", {x:L-8, y:y+4, "text-anchor":"end"}, g).textContent = v;
  }
  el("text", {x:(L+W-R)/2, y:H-6, "text-anchor":"middle"}, s).textContent = "costo della run";
  el("text", {x:14, y:(T+H-B)/2, transform:`rotate(-90 14 ${(T+H-B)/2})`, "text-anchor":"middle"}, s).textContent = "documenti esatti";
  const maxV = Math.max(...M.map(m => m.vram || 0), 1);
  const P = [...M].sort((a,b) => (b.vram||0) - (a.vram||0)).map(m => ({m, x:X(m.costo), y:Y(m.solved),
    r: m.vram ? 5 + 20 * Math.sqrt(m.vram / maxV) : 8,
    c: m.kind !== "open" ? "#fede5d" : m === migliorAperto ? "#36f9f6" : "#ff7edb"}));
  const scatole = [];
  const urta = (b) => scatole.some(q => b.x1 < q.x2 && b.x2 > q.x1 && b.y1 < q.y2 && b.y2 > q.y1) ||
    P.some(p => { const cx = Math.max(b.x1, Math.min(p.x, b.x2)), cy = Math.max(b.y1, Math.min(p.y, b.y2));
      return (cx - p.x) ** 2 + (cy - p.y) ** 2 < (p.r + 1) ** 2; }) ||
    b.x1 < L || b.x2 > W - 2 || b.y1 < 2 || b.y2 > H - B;
  for (const p of P){
    const g = el("g", {class:"punto", tabindex:0}, s);
    el("circle", {cx:p.x, cy:p.y, r:p.r, fill:p.c, "fill-opacity":.2, stroke:p.c, "stroke-width":1.5,
      "stroke-dasharray": p.m.vram ? "" : "3 3"}, g);
    el("circle", {cx:p.x, cy:p.y, r:2.5, fill:p.c}, g);
    p.g = g;
    conTip(g, () => `<b>${esc(p.m.nome)}</b><br>${p.m.solved}/${p.m.su} documenti esatti, F1 ${p.m.f1}<br>
      costo ${usd(p.m.costo)}, latenza media ${p.m.lat} s<br>${esc(p.m.gradino)}${p.m.vram ? `, circa ${p.m.vram} GB a 4 bit` : ", memoria da accertare"}`);
  }
  // etichette: prima posizione libera fra destra, sinistra, sopra, sotto, poi allontanandosi
  for (const p of P){
    const w = p.m.nome.length * 6.4 + 4, h = 14;
    let scelta = null;
    for (let passo = 0; passo < 8 && !scelta; passo++){
      const d = p.r + 5 + passo * 10;
      for (const [dx, dy, a] of [[d, 0, "start"], [-d, 0, "end"], [0, -d, "middle"], [0, d + 8, "middle"],
                                  [d * .7, -d * .7, "start"], [d * .7, d * .7, "start"], [-d * .7, -d * .7, "end"], [-d * .7, d * .7, "end"]]){
        const lx = p.x + dx, ly = p.y + dy + 4;
        const x1 = a === "start" ? lx : a === "end" ? lx - w : lx - w/2;
        const box = {x1, x2:x1 + w, y1:ly - h + 3, y2:ly + 3};
        if (!urta(box)){ scelta = {lx, ly, a, box, passo}; break; }
      }
    }
    if (!scelta) continue;
    scatole.push(scelta.box);
    if (scelta.passo > 0){
      const bx = Math.max(scelta.box.x1, Math.min(p.x, scelta.box.x2)), by = Math.max(scelta.box.y1, Math.min(p.y, scelta.box.y2));
      el("line", {x1:p.x, y1:p.y, x2:bx, y2:by, stroke:p.c, "stroke-opacity":.5}, p.g);
    }
    const t = el("text", {x:scelta.lx, y:scelta.ly, "text-anchor":scelta.a}, p.g);
    t.textContent = p.m.nome; t.style.fill = p.c;
  }
}

/* ---------- dove sbagliano ---------- */
function errori(){
  const R = ordinati.map(m => {
    const v = Object.values(D.matrice[m.id] || {});
    return {m, fn: v.reduce((s,x) => s + x.fn, 0), fp: v.reduce((s,x) => s + x.fp, 0)};
  });
  const max = Math.max(1, ...R.map(r => Math.max(r.fn, r.fp)));
  $("#errori").innerHTML = `<table><thead><tr><th>Modello</th><th class="n">Mancanti</th><th></th><th class="n">In più</th><th></th></tr></thead><tbody>` +
    R.map(r => `<tr><td>${esc(r.m.nome)}</td><td class="n">${r.fn}</td>
      <td><span class="barra"><i style="width:${90*r.fn/max}px;background:#ff7edb"></i></span></td>
      <td class="n">${r.fp}</td><td><span class="barra"><i style="width:${90*r.fp/max}px;background:#fede5d"></i></span></td></tr>`).join("") +
    `</tbody></table>`;
}

/* ---------- matrice ---------- */
function matrice(){
  const docs = D.docBench, extra = D.docSenzaCup;
  const box = $("#matrice"); box.innerHTML = "";
  box.style.gridTemplateColumns = `minmax(170px,max-content) repeat(${docs.length}, minmax(34px,1fr)) 18px` +
    (extra.length ? ` repeat(${extra.length}, minmax(28px,1fr))` : "");
  box.appendChild(document.createElement("div"));
  for (const d of docs){ const h = document.createElement("div"); h.className = "intest"; h.textContent = breve(d); h.title = d; box.appendChild(h); }
  box.appendChild(document.createElement("div"));
  for (const d of extra){ const h = document.createElement("div"); h.className = "intest"; h.style.color = "#fede5d"; h.textContent = breve(d); h.title = d; box.appendChild(h); }
  for (const m of ordinati){
    const n = document.createElement("div"); n.className = "mod";
    n.innerHTML = `${esc(m.nome)} <span style="color:var(--tenue)">${m.solved}/${m.su}</span>`;
    box.appendChild(n);
    for (const d of docs){
      const c = document.createElement("div"); const x = (D.matrice[m.id] || {})[d];
      if (!x || x.st === "assente"){ c.className = "cella vuota"; }
      else if (x.st !== "ok" && x.st !== "ok_repaired" && !x.ok){ c.className = "cella guasta"; c.textContent = "!"; }
      else if (x.ok){ c.className = "cella ris"; c.textContent = "✓"; }
      else { c.className = "cella no"; c.textContent = (x.fn ? "−"+x.fn : "") + (x.fp ? "+"+x.fp : ""); }
      if (x) conTip(c, () => {
        const righe = (lab, R) => R.map(r => `<div class="t">${lab} <code>${esc(r[0])}</code> ${eur(r[1])}${r[2] ? " PG "+esc(r[2]) : ""}</div>`).join("");
        return `<b>${esc(m.nome)}</b> su ${esc(d)}<br>benchmark ${x.gt} righe, modello ${x.pred}, stato cella: ${esc(x.st)}` +
          (x.ok ? "<br>identico al benchmark" : righe("manca", x.fnR) + righe("in più", x.fpR) +
            ((x.fn > x.fnR.length || x.fp > x.fpR.length) ? `<div class="t">(elenco troncato)</div>` : ""));
      });
      box.appendChild(c);
    }
    box.appendChild(document.createElement("div"));
    for (const d of extra){
      const c = document.createElement("div");
      const cups = [...new Set(D.rec.filter(r => r.m === m.id && r.d === d).map(r => r.cup))];
      c.className = "cella " + (cups.length ? "inv" : "pulita");
      c.textContent = cups.length ? cups.length : "·";
      conTip(c, () => cups.length
        ? `<b>${esc(m.nome)}</b> su ${esc(d)}<br>${cups.length} CUP trovati in un atto senza CUP nel benchmark:<div class="t">${cups.slice(0,10).map(esc).join(", ")}</div>`
        : `<b>${esc(m.nome)}</b> su ${esc(d)}<br>nessun CUP estratto, corretto`);
      box.appendChild(c);
    }
  }
}

/* ---------- flusso PG verso CUP ---------- */
const fFonte = $("#fFonte"), fDoc = $("#fDoc"), fStato = $("#fStato"), fNoPg = $("#fNoPg");
function opzioni(sel, voci){ sel.innerHTML = voci.map(([v,t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join(""); }
function impostaFiltri(){
  opzioni(fFonte, [["consenso", `Lettura condivisa (almeno ${soglia} modelli aperti su ${aperti.length})`],
    ...ordinati.map(m => [m.id, m.nome + (m.kind==="open" ? "" : " (commerciale)")]), ["benchmark", "Benchmark di Alessandro"]]);
  const tuttiDoc = [...new Set(D.rec.map(r => r.d))].sort();
  opzioni(fDoc, [["", "Tutti i documenti"], ...tuttiDoc.map(d => [d, d])]);
  const stati = [...new Set(D.rec.map(r => r.stato))].sort();
  opzioni(fStato, [["", "Tutti gli stati"], ...stati.map(s => [s, s])]);
}
$("#legStati").innerHTML = Object.entries(COL_STATO).map(([s,c]) => `<span><i style="background:${c}"></i>${s === "altro" ? "altri stati" : s}</span>`).join("");

function recordFonte(){
  const f = fFonte.value;
  let R = f === "consenso" ? consenso
    : f === "benchmark" ? D.gt.map(g => ({m:"benchmark", d:g.d, cup:g.cup, imp:g.imp, pg:g.pg,
        cap:g.pg ? g.pg.split("/")[0] : null, stato:g.stato || "altro", liv:"totale", rif:"Output a MANO", txt:""}))
    : D.rec.filter(r => r.m === f);
  if (fDoc.value) R = R.filter(r => r.d === fDoc.value);
  if (fStato.value) R = R.filter(r => r.stato === fStato.value);
  R = R.filter(r => r.liv !== "quota");
  const senza = R.filter(r => !r.pg).length;
  if (!fNoPg.checked) R = R.filter(r => r.pg);
  return {R, senza};
}

const LIMITE_CUP = 220;
function flusso(){
  const {R, senza} = recordFonte();
  const svg = $("#flusso"); svg.innerHTML = ""; svg.classList.remove("fuoco");
  const nd = "non indicato";
  // aggregazione
  const cupTot = new Map();
  for (const r of R) cupTot.set(r.cup, (cupTot.get(r.cup) || 0) + Math.abs(r.imp));
  let cups = [...cupTot.entries()].sort((a,b) => b[1] - a[1]);
  const tagliati = Math.max(0, cups.length - LIMITE_CUP);
  cups = cups.slice(0, LIMITE_CUP);
  const tieni = new Set(cups.map(c => c[0]));
  const RR = R.filter(r => tieni.has(r.cup));
  const eCP = new Map(), ePC = new Map(), pgTot = new Map(), capTot = new Map();
  for (const r of RR){
    const cap = r.cap || nd, pg = r.pg || nd, v = Math.abs(r.imp);
    const k1 = cap + "§" + pg, k2 = pg + "§" + r.cup;
    if (!eCP.has(k1)) eCP.set(k1, {da:cap, a:pg, v:0, R:[]});
    if (!ePC.has(k2)) ePC.set(k2, {da:pg, a:r.cup, v:0, R:[]});
    eCP.get(k1).v += v; eCP.get(k1).R.push(r);
    ePC.get(k2).v += v; ePC.get(k2).R.push(r);
    pgTot.set(pg, (pgTot.get(pg) || 0) + v); capTot.set(cap, (capTot.get(cap) || 0) + v);
  }
  const totale = RR.reduce((s,r) => s + Math.abs(r.imp), 0);
  $("#sintesi").innerHTML = RR.length
    ? `<span><b>${tieni.size}</b> CUP</span><span><b>${[...pgTot.keys()].filter(p => p !== nd).length}</b> piani gestionali</span>
       <span><b>${[...capTot.keys()].filter(c => c !== nd).length}</b> capitoli</span><span><b>${eurBreve(totale)}</b> come somma degli eventi</span>
       ${!fNoPg.checked && senza ? `<span>${senza} record senza PG esclusi</span>` : ""}
       ${tagliati ? `<span class="avviso">mostro i ${LIMITE_CUP} CUP con importo maggiore, ${tagliati} esclusi</span>` : ""}`
    : "";
  if (!RR.length){
    svg.setAttribute("viewBox", "0 0 1100 120");
    const t = el("text", {x:550, y:60, "text-anchor":"middle"}, svg);
    t.textContent = senza ? `Nessun record con PG per questa selezione: ${senza} record non indicano il piano gestionale. Spunta «includi i record senza PG».` : "Nessun record per questa selezione.";
    return;
  }
  // disposizione: CUP ordinati per PG principale, PG e capitoli al baricentro
  const pgPrinc = new Map();
  for (const e of ePC.values()){
    const p = pgPrinc.get(e.a); if (!p || e.v > p.v) pgPrinc.set(e.a, {pg:e.da, v:e.v});
  }
  const pgOrd = [...pgTot.keys()].sort((a,b) => (a===nd) - (b===nd) || a.localeCompare(b, "it", {numeric:true}));
  const cupOrd = cups.map(c => c[0]).sort((a,b) =>
    pgOrd.indexOf(pgPrinc.get(a).pg) - pgOrd.indexOf(pgPrinc.get(b).pg) || cupTot.get(b) - cupTot.get(a));
  const W = 1100, T = 44, PASSO = 30, H = Math.max(260, T + cupOrd.length * PASSO + 20);
  const xCap = 70, xPg = 470, xCup = 850, lN = 110;
  const y = new Map();
  cupOrd.forEach((c,i) => y.set("u:"+c, T + 10 + i * PASSO + PASSO/2));
  function baricentro(chiavi, archi, pref, dest){
    const pos = chiavi.map(k => {
      let s = 0, w = 0;
      for (const e of archi) if (e.da === k){ s += y.get(dest + e.a) * e.v; w += e.v; }
      return {k, y: w ? s / w : T};
    }).sort((a,b) => a.y - b.y);
    const gap = 34;
    for (let i = 1; i < pos.length; i++) if (pos[i].y < pos[i-1].y + gap) pos[i].y = pos[i-1].y + gap;
    const eccesso = pos.length ? pos[pos.length-1].y - (H - 24) : 0;
    if (eccesso > 0) for (const p of pos) p.y -= eccesso;
    for (let i = 0; i < pos.length; i++) if (pos[i].y < T + 14) pos[i].y = T + 14 + i * gap;
    pos.forEach(p => y.set(pref + p.k, p.y));
  }
  baricentro(pgOrd, [...ePC.values()], "p:", "u:");
  baricentro([...capTot.keys()], [...eCP.values()], "c:", "p:");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.style.height = "auto";

  // definizioni: bagliore
  const defs = el("defs", {}, svg);
  const f = el("filter", {id:"glow", x:"-20%", y:"-50%", width:"140%", height:"200%"}, defs);
  el("feGaussianBlur", {stdDeviation:"3", result:"b"}, f);
  const mg = el("feMerge", {}, f); el("feMergeNode", {in:"b"}, mg); el("feMergeNode", {in:"SourceGraphic"}, mg);
  const lg = el("linearGradient", {id:"gCap", gradientUnits:"userSpaceOnUse", x1:xCap + lN, x2:xPg, y1:0, y2:0}, defs);
  el("stop", {offset:"0", "stop-color":"#848bbd"}, lg); el("stop", {offset:"1", "stop-color":"#ff7edb"}, lg);

  [["Capitolo", xCap], ["Piano gestionale", xPg], ["CUP", xCup]].forEach(([t,x]) => {
    const n = el("text", {x: x + lN/2, y:22, "text-anchor":"middle", class:"colonna"}, svg); n.textContent = t;
  });

  const vMax = Math.max(...[...ePC.values(), ...eCP.values()].map(e => e.v));
  const spess = v => 1.5 + 16 * Math.sqrt(v / vMax);
  const gArchi = el("g", {}, svg), gNodi = el("g", {}, svg);
  const collegati = new Map();
  const lega = (a, b) => { (collegati.get(a) || collegati.set(a, new Set()).get(a)).add(b); };

  function arco(e, x1, k1, x2, k2, colore, tipTesto){
    const y1 = y.get(k1), y2 = y.get(k2), w = spess(e.v), punta = Math.max(9, w * 0.9);
    const xe = x2 - punta, mx = (x1 + xe) / 2;
    const d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${xe},${y2}`;
    const g = el("g", {class:"grp"}, gArchi);
    const p = el("path", {d, class:"arco", stroke:colore, "stroke-width":w, "stroke-opacity":.72, filter:"url(#glow)"}, g);
    const tri = el("path", {class:"punta", d:`M${xe-1},${y2 - punta*0.75} L${x2},${y2} L${xe-1},${y2 + punta*0.75} Z`, fill:colore}, g);
    el("path", {d, class:"scia", "stroke-width":Math.max(1, w * 0.35)}, g);
    const id = k1 + ">" + k2;
    g.dataset.id = id; lega(k1, id); lega(k2, id);
    conTip(p, tipTesto); conTip(tri, tipTesto);
    return g;
  }
  const dominante = rs => { const c = {}; for (const r of rs) c[r.stato] = (c[r.stato] || 0) + Math.abs(r.imp);
    return Object.entries(c).sort((a,b) => b[1]-a[1])[0][0]; };
  const elenco = rs => rs.slice(0,4).map(r => `<div class="t">${esc(breve(r.d))}: ${eur(r.imp)}, ${esc(r.stato)}${r.rif ? ", " + esc(r.rif) : ""}${r.txt ? `<br><i>${esc(r.txt.slice(0,140))}</i>` : ""}</div>`).join("") + (rs.length > 4 ? `<div class="t">e altri ${rs.length-4} record</div>` : "");

  for (const e of eCP.values())
    arco(e, xCap + lN, "c:"+e.da, xPg, "p:"+e.a, "url(#gCap)",
      () => `<b>Capitolo ${esc(e.da)} verso PG ${esc(e.a)}</b><br>${eur(e.v)}`);
  for (const e of ePC.values()){
    const st = dominante(e.R);
    arco(e, xPg + lN, "p:"+e.da, xCup, "u:"+e.a, colStato(st),
      () => `<b>PG ${esc(e.da)} verso ${esc(e.a)}</b><br>${eur(e.v)}, stato prevalente ${esc(st)}${elenco(e.R)}`);
  }

  function nodo(k, x, testo, sub, colore, extra){
    const g = el("g", {class:"nodo", tabindex:0, "data-id":k}, gNodi);
    const yy = y.get(k);
    el("rect", {x, y:yy - 12, width:lN, height:24, fill:"#1b1526", stroke:colore}, g);
    const t = el("text", {x:x + 8, y:yy + 4}, g); t.textContent = testo;
    if (sub){ const s = el("text", {x: extra === "dx" ? x + lN + 8 : x - 8, y:yy + 4, class:"sub", "text-anchor": extra === "dx" ? "start" : "end"}, g); s.textContent = sub; }
    g.addEventListener("mouseenter", () => evidenzia(k));
    g.addEventListener("mouseleave", spegni);
    g.addEventListener("focus", () => evidenzia(k));
    g.addEventListener("blur", spegni);
    return g;
  }
  for (const [c,v] of capTot) nodo("c:"+c, xCap, c === nd ? "n.d." : c, eurBreve(v), "#848bbd");
  for (const [p,v] of pgTot){
    const g = nodo("p:"+p, xPg, p === nd ? "n.d." : `${p.split("/")[0]} pg ${p.split("/")[1] || "?"}`, eurBreve(v), "#ff7edb");
    const s = g.querySelector(".sub");  // sotto la pillola, lontano dalle punte delle frecce
    s.setAttribute("x", xPg + lN/2); s.setAttribute("y", y.get("p:"+p) + 26); s.setAttribute("text-anchor", "middle");
  }
  for (const c of cupOrd){
    const g = nodo("u:"+c, xCup, c, eurBreve(cupTot.get(c)), "#36f9f6", "dx");
    g.querySelector("rect").setAttribute("width", 140);
    g.querySelector(".sub").setAttribute("x", xCup + 148);
    g.addEventListener("click", () => scheda(c));
    g.addEventListener("keydown", ev => { if (ev.key === "Enter") scheda(c); });
  }

  function evidenzia(k){
    // la catena intera: dal CUP risale al PG e al capitolo, dal capitolo scende ai CUP
    const acceso = new Set([k]);
    const espandi = (n, verso) => {
      for (const id of collegati.get(n) || []){
        const [a, b] = id.split(">");
        if (verso === "giu" && a === n){ acceso.add(id); acceso.add(b); if (!b.startsWith("u:")) espandi(b, "giu"); }
        if (verso === "su" && b === n){ acceso.add(id); acceso.add(a); if (!a.startsWith("c:")) espandi(a, "su"); }
      }
    };
    espandi(k, "giu"); espandi(k, "su");
    svg.classList.add("fuoco");
    svg.querySelectorAll(".grp,.nodo").forEach(n => n.classList.toggle("caldo", acceso.has(n.dataset.id)));
  }
  function spegni(){ svg.classList.remove("fuoco"); svg.querySelectorAll(".caldo").forEach(n => n.classList.remove("caldo")); }
}
[fFonte, fDoc, fStato, fNoPg].forEach(c => c.addEventListener("change", flusso));

/* ---------- scheda CUP ---------- */
function scheda(cup){
  const box = $("#scheda");
  const bench = D.gt.filter(g => g.cup === cup);
  const perEvento = new Map();
  for (const r of D.rec.filter(r => r.cup === cup)){
    const k = [r.d, r.pg || "", Math.round(r.imp), r.stato].join("|");
    if (!perEvento.has(k)) perEvento.set(k, {r, mod:new Set()});
    perEvento.get(k).mod.add(nome(r.m));
  }
  const righe = [...perEvento.values()].sort((a,b) => a.r.d.localeCompare(b.r.d) || b.mod.size - a.mod.size);
  const nModTot = new Set(D.rec.filter(r => r.cup === cup).map(r => r.m)).size;
  box.innerHTML = `<div class="scheda" tabindex="-1">
    <button class="chiudi" type="button">Chiudi</button>
    <button class="chiudi aggT" type="button" style="margin-right:8px">Aggiungi alla linea del tempo</button>
    <h3>${esc(cup)}</h3>
    <p class="nota">Trovato da ${nModTot} modelli su ${M.length}. ${bench.length ? `Nel benchmark: ${bench.map(g => `${esc(breve(g.d))} ${eur(g.imp)}${g.pg ? " PG " + esc(g.pg) : ""}`).join("; ")}.` : "Non compare nel benchmark."}</p>
    <div class="scorri"><table><thead><tr><th>Documento</th><th>PG</th><th class="n">Importo</th><th>Stato</th><th>Livello</th><th>Fonte nel testo</th><th class="n">Modelli</th></tr></thead>
    <tbody>${righe.map(({r, mod}) => `<tr title="${esc([...mod].join(", "))}"><td>${esc(breve(r.d))}</td><td>${esc(r.pg || "n.d.")}</td>
      <td class="n">${eur(r.imp)}</td><td><span style="color:${colStato(r.stato)}">${esc(r.stato)}</span></td><td>${esc(r.liv)}</td>
      <td style="white-space:normal;max-width:360px">${esc(r.rif)}</td><td class="n">${mod.size}</td></tr>`).join("")}</tbody></table></div></div>`;
  box.querySelector(".chiudi:not(.aggT)").onclick = () => box.innerHTML = "";
  box.querySelector(".aggT").onclick = () => { aggiungiCup(cup); $("#sezTempo").scrollIntoView({behavior:"smooth"}); };
  box.firstElementChild.focus({preventScroll:true});
  box.scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block:"nearest"});
}

/* ---------- elenco CUP e PG ---------- */
let linguetta = "cup";
let ordE = {k:"imp", dir:-1};
function elenco(){
  const q = $("#cerca").value.trim().toUpperCase();
  const t = $("#elenco");
  const ord = (A) => A.sort((a,b) => { const x = a[ordE.k], y = b[ordE.k];
    return (typeof x === "string" ? x.localeCompare(y) : (x - y)) * ordE.dir; });
  const th = (k, t, n) => `<th class="${n?"n":""}"><button data-k="${k}" ${ordE.k===k?`aria-sort="${ordE.dir>0?"ascending":"descending"}"`:""}>${t}</button></th>`;
  if (linguetta === "cup"){
    const R = ord(cupRighe.filter(x => !q || x.cup.includes(q) || [...x.docs].some(d => d.toUpperCase().includes(q)) || [...x.pg].some(p => p.includes(q))));
    t.innerHTML = `<thead><tr>${th("cup","CUP")}${th("nMod","Modelli",1)}<th>Benchmark</th>${th("imp","Importo condiviso",1)}<th>PG</th><th>Documenti</th></tr></thead>
      <tbody>${R.map(x => `<tr class="cliccabile" data-cup="${esc(x.cup)}" tabindex="0"><td><code>${esc(x.cup)}</code></td><td class="n">${x.nMod}/${M.length}</td>
      <td>${x.bench ? '<span class="ok">sì</span>' : '<span style="color:var(--tenue)">no</span>'}</td>
      <td class="n">${x.imp ? eur(x.imp) : '<span style="color:var(--tenue)">nessun accordo</span>'}</td>
      <td>${[...x.pg].map(esc).join(", ") || "n.d."}</td><td style="white-space:normal">${[...x.docs].map(d => esc(breve(d))).join(", ")}</td></tr>`).join("")}</tbody>`;
    t.querySelectorAll("tr[data-cup]").forEach(r => { r.onclick = () => scheda(r.dataset.cup); r.onkeydown = e => { if (e.key === "Enter") scheda(r.dataset.cup); }; });
  } else {
    const R = ord(pgRighe.map(x => ({...x, nCup:x.cups.size}))
      .filter(x => !q || x.pg.includes(q) || [...x.cups].some(c => c.includes(q)) || [...x.docs].some(d => d.toUpperCase().includes(q))));
    t.innerHTML = `<thead><tr>${th("pg","Capitolo / PG")}${th("nCup","CUP finanziati",1)}${th("imp","Importo condiviso",1)}${th("nMod","Modelli che lo citano",1)}<th>Stati</th><th>Documenti</th></tr></thead>
      <tbody>${R.map(x => `<tr><td><code>${esc(x.pg)}</code></td><td class="n">${x.cups.size}</td><td class="n">${x.imp ? eur(x.imp) : "nessun accordo"}</td>
      <td class="n">${x.nMod}/${M.length}</td><td>${Object.entries(x.stati).map(([s,n]) => `<span style="color:${colStato(s)}">${esc(s)}</span> ${n}`).join(", ")}</td>
      <td style="white-space:normal">${[...x.docs].map(d => esc(breve(d))).join(", ")}</td></tr>`).join("")}</tbody>`;
  }
  t.querySelectorAll("th button").forEach(b => b.onclick = () => {
    const k = b.dataset.k === "nCup" ? "nCup" : b.dataset.k;
    ordE = {k, dir: ordE.k === k ? -ordE.dir : (k === "cup" || k === "pg" ? 1 : -1)}; elenco(); });
}
document.querySelectorAll(".linguette button").forEach(b => b.onclick = () => {
  linguetta = b.dataset.t;
  document.querySelectorAll(".linguette button").forEach(x => x.setAttribute("aria-selected", x === b));
  ordE = {k:"imp", dir:-1}; elenco();
});
$("#cerca").addEventListener("input", elenco);

/* ---------- linea del tempo ---------- */
const normDoc = s => String(s).toLowerCase().replace(/\.pdf$/, "").replace(/[^a-z0-9]/g, "");
const dataDoc = d => (INDICE.date || {})[normDoc(d)] || null;
const MESI_B = ["gen","feb","mar","apr","mag","giu","lug","ago","set","ott","nov","dic"];
const giorno = iso => { const [y,m,d] = iso.split("-").map(Number); return new Date(Date.UTC(y, m-1, d)); };
const fmtData = iso => { const t = giorno(iso); return `${t.getUTCDate()} ${MESI_B[t.getUTCMonth()]} ${t.getUTCFullYear()}`; };
const AFF = {certa:"certa", nome:"dal nome del file", stimata:"stimata"};
let cupScelti = [];
const tFonte = $("#tFonte");

function recordTempo(){
  const f = tFonte.value;
  const R = f === "consenso" ? consenso
    : f === "benchmark" ? D.gt.map(g => ({m:"benchmark", d:g.d, cup:g.cup, imp:g.imp, pg:g.pg, stato:g.stato || "altro", liv:"totale", rif:"Output a MANO", txt:""}))
    : D.rec.filter(r => r.m === f);
  return R.filter(r => r.liv !== "quota");
}
function aggiungiCup(c){
  c = String(c || "").trim().toUpperCase();
  if (!c || cupScelti.includes(c)) return;
  cupScelti.push(c); tempo();
}
function impostaTempo(){
  const v = tFonte.value;
  opzioni(tFonte, [["consenso", "Lettura condivisa dei modelli aperti"],
    ...ordinati.map(m => [m.id, m.nome]), ["benchmark", "Benchmark di Alessandro"]]);
  if ([...tFonte.options].some(o => o.value === v)) tFonte.value = v;
  const cups = [...new Set(D.rec.map(r => r.cup))].sort();
  $("#tCupElenco").innerHTML = cups.map(c => `<option value="${esc(c)}">`).join("");
  if (!cupScelti.length) cupScelti = piuAttivi(3);
}
function piuAttivi(n){
  const conta = new Map();
  for (const r of recordTempo()){ if (!conta.has(r.cup)) conta.set(r.cup, new Set()); conta.get(r.cup).add(r.d); }
  return [...conta.entries()].sort((a,b) => b[1].size - a[1].size || a[0].localeCompare(b[0])).slice(0, n).map(x => x[0]);
}
function tempo(){
  const svg = $("#tempo"); svg.innerHTML = "";
  $("#tScelti").innerHTML = cupScelti.map(c => `<span class="chip">${esc(c)}<button type="button" data-c="${esc(c)}" aria-label="Togli ${esc(c)}">×</button></span>`).join("")
    || `<span style="color:var(--tenue)">Nessun CUP scelto.</span>`;
  $("#tScelti").querySelectorAll("button").forEach(b => b.onclick = () => { cupScelti = cupScelti.filter(x => x !== b.dataset.c); tempo(); });

  const R = recordTempo().filter(r => cupScelti.includes(r.cup));
  // un evento per (CUP, atto): gli importi dello stesso atto si raccolgono in un cerchio
  const ev = new Map(), senzaData = new Set();
  for (const r of R){
    const dd = dataDoc(r.d);
    if (!dd){ senzaData.add(r.d); continue; }
    const k = r.cup + "|" + r.d;
    if (!ev.has(k)) ev.set(k, {cup:r.cup, d:r.d, data:dd, R:[], v:0});
    const e = ev.get(k); e.R.push(r); e.v += Math.abs(r.imp);
  }
  const E = [...ev.values()];
  // asse: tutti gli atti datati del corpus eseguito, cosi' la scala resta stabile fra CUP
  const attiRun = [...new Set(D.rec.map(r => r.d).concat(D.docBench, D.docSenzaCup))].map(d => ({d, dd:dataDoc(d)})).filter(x => x.dd);
  const tutteDate = attiRun.flatMap(x => [x.dd.min || x.dd.data, x.dd.max || x.dd.data]).map(giorno);
  if (!tutteDate.length){ svg.setAttribute("viewBox","0 0 1100 80"); el("text",{x:550,y:44,"text-anchor":"middle"},svg).textContent = "Nessuna data disponibile per gli atti di questa run."; cronologia([]); return; }
  let t0 = new Date(Math.min(...tutteDate)), t1 = new Date(Math.max(...tutteDate));
  t0 = new Date(Date.UTC(t0.getUTCFullYear(), t0.getUTCMonth() - 1, 1));
  t1 = new Date(Date.UTC(t1.getUTCFullYear(), t1.getUTCMonth() + 2, 1));
  const W = 1100, L = 170, Rm = 30, TOP = 58, CORSIA = 64;
  const H = TOP + Math.max(1, cupScelti.length) * CORSIA + 30;
  const X = d => L + (d - t0) / (t1 - t0) * (W - L - Rm);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.style.height = "auto";

  // anni e mesi
  const ga = el("g", {}, svg);
  for (let y = t0.getUTCFullYear(); y <= t1.getUTCFullYear(); y++){
    for (let m = 0; m < 12; m++){
      const d = new Date(Date.UTC(y, m, 1)); if (d < t0 || d > t1) continue;
      const x = X(d);
      el("line", {x1:x, x2:x, y1:TOP-12, y2:H-24, stroke: m === 0 ? "#ff7edb" : "#463a63", "stroke-opacity": m === 0 ? .45 : .35, "stroke-dasharray": m === 0 ? "" : "1 5"}, ga);
      if (m === 0) el("text", {x:x+4, y:TOP-18, class:"anno"}, ga).textContent = y;
      else if (m % 3 === 0) el("text", {x, y:H-8, "text-anchor":"middle", class:"etic"}, ga).textContent = MESI_B[m];
    }
  }
  // tacche degli atti in alto: dove cadono tutti i decreti della run
  for (const a of attiRun){
    const x = X(giorno(a.dd.data));
    const tk = el("line", {x1:x, x2:x, y1:TOP-8, y2:TOP-2, class:"atto", "stroke-width":2}, svg);
    conTip(tk, () => `<b>${esc(a.d)}</b><br>${fmtData(a.dd.data)} (${AFF[a.dd.aff] || a.dd.aff})`);
  }

  const vMax = Math.max(1, ...E.map(e => e.v));
  cupScelti.forEach((c, i) => {
    const y0 = TOP + i * CORSIA, yc = y0 + CORSIA / 2;
    el("rect", {x:L-6, y:y0+6, width:W-L-Rm+12, height:CORSIA-12, rx:8, class:"corsia"}, svg);
    const qui = E.filter(e => e.cup === c).sort((a,b) => giorno(a.data.data) - giorno(b.data.data));
    const lab = el("text", {x:L-14, y:yc-2, "text-anchor":"end"}, svg); lab.textContent = c; lab.style.fill = "#f4eee4"; lab.style.fontWeight = 600;
    const sub = el("text", {x:L-14, y:yc+14, "text-anchor":"end", class:"etic"}, svg);
    sub.textContent = qui.length ? `${qui.length} ${qui.length === 1 ? "atto" : "atti"}` : "nessun atto con data";
    if (qui.length > 1) el("line", {x1:X(giorno(qui[0].data.data)), x2:X(giorno(qui[qui.length-1].data.data)), y1:yc, y2:yc, class:"vita"}, svg);
    qui.forEach((e, j) => {
      const dd = e.data, x = X(giorno(dd.data));
      const pesi = {}; for (const r of e.R) pesi[r.stato] = (pesi[r.stato] || 0) + Math.abs(r.imp);
      const st = Object.entries(pesi).sort((a,b) => b[1]-a[1])[0][0], col = colStato(st);
      const g = el("g", {class:"evento", tabindex:0}, svg);
      if (dd.aff === "stimata" && dd.min && dd.max)
        el("line", {x1:X(giorno(dd.min)), x2:X(giorno(dd.max)), y1:yc, y2:yc, stroke:col, class:"forchetta"}, g);
      const r = 5 + 13 * Math.sqrt(e.v / vMax);
      el("circle", {cx:x, cy:yc, r, fill: dd.aff === "stimata" ? "#1b1526" : col, "fill-opacity": dd.aff === "stimata" ? 1 : .85,
        stroke:col, "stroke-width":2, "stroke-dasharray": dd.aff === "stimata" ? "3 2" : ""}, g);
      const lt = el("text", {x, y: j % 2 ? yc + r + 12 : yc - r - 5, "text-anchor":"middle", class:"etic"}, g);
      lt.textContent = eurBreve(e.v);
      conTip(g, () => `<b>${esc(e.cup)}</b>, ${esc(breve(e.d))}<br>${fmtData(dd.data)}, data ${AFF[dd.aff] || dd.aff}${dd.aff === "stimata" && dd.min ? ` (fra ${fmtData(dd.min)} e ${fmtData(dd.max)})` : ""}` +
        e.R.slice(0,5).map(r => `<div class="t"><span style="color:${colStato(r.stato)}">${esc(r.stato)}</span> ${eur(r.imp)}${r.pg ? ", PG " + esc(r.pg) : ""}${r.rif ? ", " + esc(r.rif) : ""}</div>`).join("") +
        (e.R.length > 5 ? `<div class="t">e altri ${e.R.length - 5} record</div>` : ""));
      g.addEventListener("click", () => scheda(e.cup));
    });
  });
  $("#tLegenda").innerHTML = Object.entries(COL_STATO).map(([s,c]) => `<span><i style="background:${c}"></i>${s === "altro" ? "altri stati" : s}</span>`).join("") +
    `<span><i style="border:2px dashed #848bbd;border-radius:50%"></i>data stimata, con forchetta</span>` +
    (senzaData.size ? `<span class="avviso">${senzaData.size} atti senza data esclusi</span>` : "");
  cronologia(E);
}
function cronologia(E){
  const righe = E.flatMap(e => e.R.map(r => ({e, r}))).sort((a,b) => giorno(a.e.data.data) - giorno(b.e.data.data) || a.r.cup.localeCompare(b.r.cup));
  $("#tCrono").innerHTML = `<thead><tr><th>Data</th><th>CUP</th><th>Atto</th><th>Stato</th><th>PG</th><th class="n">Importo</th><th>Fonte nel testo</th></tr></thead><tbody>` +
    (righe.length ? righe.map(({e, r}) => `<tr><td>${fmtData(e.data.data)} <span class="aff ${esc(e.data.aff)}">${esc(AFF[e.data.aff] || e.data.aff)}</span></td>
      <td><code>${esc(r.cup)}</code></td><td>${esc(breve(r.d))}</td><td><span style="color:${colStato(r.stato)}">${esc(r.stato)}</span></td>
      <td>${esc(r.pg || "n.d.")}</td><td class="n">${eur(r.imp)}</td><td style="white-space:normal;max-width:380px">${esc(r.rif || "")}</td></tr>`).join("")
     : `<tr><td colspan="7" style="color:var(--tenue)">Nessun evento per i CUP scelti con questa fonte.</td></tr>`) + `</tbody>`;
}
function tabellaDate(){
  const atti = [...new Set(D.rec.map(r => r.d).concat(D.docBench, D.docSenzaCup))].sort((a,b) => {
    const x = dataDoc(a), y = dataDoc(b); return (x ? x.data : "9") < (y ? y.data : "9") ? -1 : 1; });
  $("#tDate").innerHTML = `<thead><tr><th>Atto</th><th>Data</th><th>Affidabilità</th><th>Come è stata ricavata</th></tr></thead><tbody>` +
    atti.map(d => { const x = dataDoc(d);
      return `<tr><td>${esc(d)}</td><td>${x ? fmtData(x.data) : "sconosciuta"}${x && x.min ? `<div class="t" style="color:var(--tenue);font-size:13px">fra ${fmtData(x.min)} e ${fmtData(x.max)}</div>` : ""}</td>
        <td>${x ? `<span class="aff ${esc(x.aff)}">${esc(AFF[x.aff] || x.aff)}</span>` : ""}</td><td style="white-space:normal;max-width:560px">${x ? esc(x.fonte) : "aggiungila in date_decreti.csv"}</td></tr>`; }).join("") + `</tbody>`;
}
tFonte.addEventListener("change", tempo);
$("#tAggiungi").onclick = () => { aggiungiCup($("#tCup").value); $("#tCup").value = ""; };
$("#tCup").addEventListener("keydown", e => { if (e.key === "Enter"){ aggiungiCup(e.target.value); e.target.value = ""; } });
$("#tPiuAttivi").onclick = () => { cupScelti = piuAttivi(6); tempo(); };
$("#tSvuota").onclick = () => { cupScelti = []; tempo(); };

/* ---------- storico, scelta della run, prompt ---------- */
const selRun = $("#selRun");
selRun.innerHTML = [...INDICE.runs].reverse().map(r =>
  `<option value="${esc(r.run)}">${esc(r.dataBreve)}, ${esc(r.versione || "prompt " + r.fp)}, ${r.nDoc} documenti, ${r.nModelli} modelli${r.miglioreAperto ? `, meglio ${esc(r.miglioreAperto.nome)} ${r.miglioreAperto.solved}/${r.miglioreAperto.su}` : ""}</option>`).join("");
selRun.addEventListener("change", () => caricaRun(selRun.value));
window.ECHO_RUN = d => { CACHE[d.meta.run] = d; };

function caricaRun(nomeRun){
  if (CACHE[nomeRun]) return mostraRun(CACHE[nomeRun]);
  const sc = document.createElement("script");
  sc.src = "cruscotto_dati/" + encodeURIComponent(nomeRun) + ".js";
  sc.onload = () => CACHE[nomeRun] ? mostraRun(CACHE[nomeRun]) : errore(nomeRun);
  sc.onerror = () => errore(nomeRun);
  document.head.appendChild(sc);
}
function errore(nomeRun){
  $("#titolo").textContent = `Non trovo i dati della run ${nomeRun}: rilancia python dashboard.py.`;
}
function mostraRun(d){
  D = d; prepara();
  selRun.value = D.meta.run;
  history.replaceState(null, "", "#run=" + encodeURIComponent(D.meta.run));
  testata(); classifica(); scatter(); errori(); matrice();
  impostaFiltri(); flusso(); $("#scheda").innerHTML = "";
  impostaTempo(); tempo(); tabellaDate();
  ordE = {k:"imp", dir:-1}; elenco();
  document.querySelectorAll("#tabRun tr[data-run]").forEach(r => r.classList.toggle("corrente", r.dataset.run === D.meta.run));
  $("#piede").textContent = `Generato da dashboard.py il ${INDICE.generato}. La somma degli eventi non è una disponibilità: lo stesso denaro può comparire in più atti (assegnazione, poi impegno).`;
}

function storico(){
  const R = INDICE.runs, s = $("#evoluzione"), W = 1100, H = 270, L = 90, Rm = 60, T = 14, B = 58;
  s.innerHTML = "";
  if (!R.length) return;
  const X = i => R.length === 1 ? (L + W - Rm) / 2 : L + i * (W - L - Rm) / (R.length - 1);
  const Y = v => T + (1 - v) * (H - T - B);
  const g = el("g", {class:"asse"}, s);
  for (const v of [0, .25, .5, .75, 1]){
    el("line", {x1:L, x2:W-Rm, y1:Y(v), y2:Y(v), "stroke-dasharray":"2 4"}, g);
    el("text", {x:L-8, y:Y(v)+4, "text-anchor":"end"}, g).textContent = Math.round(v*100) + "%";
  }
  const passo = Math.max(1, Math.ceil(R.length / 12));
  R.forEach((r, i) => {
    if (i % passo) return;
    const t1 = el("text", {x:X(i), y:H-B+18, "text-anchor":"middle"}, s); t1.textContent = r.dataBreve;
    const t2 = el("text", {x:X(i), y:H-B+34, "text-anchor":"middle"}, s); t2.textContent = r.versione || r.fp; t2.style.fill = "#ff7edb";
  });
  const linea = R.map((r, i) => r.miglioreAperto && r.miglioreAperto.su ? [X(i), Y(r.miglioreAperto.solved / r.miglioreAperto.su)] : null).filter(Boolean);
  if (linea.length > 1) el("polyline", {points:linea.map(p => p.join(",")).join(" "), fill:"none", stroke:"#36f9f6", "stroke-width":2, "stroke-opacity":.8}, s);
  R.forEach((r, i) => {
    for (const m of r.modelli){
      if (!m.su) continue;
      const best = r.miglioreAperto && m.nome === r.miglioreAperto.nome;
      const colore = m.kind !== "open" ? "#fede5d" : best ? "#36f9f6" : "#ff7edb";
      const c = el("circle", {cx:X(i) + (best ? 0 : ((r.modelli.indexOf(m) % 5) - 2) * 3), cy:Y(m.solved / m.su), r: best ? 6 : 4,
        fill:colore, "fill-opacity": best ? 1 : .55, stroke: best ? "#fff" : "none", "stroke-width":1.5}, s);
      c.style.cursor = "pointer";
      c.addEventListener("click", () => caricaRun(r.run));
      conTip(c, () => `<b>${esc(m.nome)}</b><br>${m.solved}/${m.su} documenti esatti, F1 ${m.f1}<div class="t">run del ${esc(r.dataEstesa)}, prompt ${esc(r.versione || r.fp)}</div>`);
    }
  });
  $("#tabRun").innerHTML = `<thead><tr><th>Data</th><th>Prompt</th><th class="n">Documenti</th><th class="n">Modelli</th><th class="n">Spesa</th>
    <th>Miglior aperto</th><th class="n">Esatti</th><th>Miglior assoluto</th><th>Cartella</th></tr></thead><tbody>` +
    [...R].reverse().map(r => `<tr class="cliccabile" data-run="${esc(r.run)}" tabindex="0"><td>${esc(r.dataEstesa)}</td>
      <td><button class="linkp" data-fp="${esc(r.fp)}" type="button">${esc(r.versione || "senza versione")} (${esc(r.fp)})</button></td>
      <td class="n">${r.nDoc}</td><td class="n">${r.nModelli}</td><td class="n">${usd(r.costo)}</td>
      <td>${r.miglioreAperto ? esc(r.miglioreAperto.nome) : "nessuno"}</td>
      <td class="n">${r.miglioreAperto ? `${r.miglioreAperto.solved}/${r.miglioreAperto.su}` : ""}</td>
      <td>${r.miglioreTot ? `${esc(r.miglioreTot.nome)} ${r.miglioreTot.solved}/${r.miglioreTot.su}` : ""}</td>
      <td style="color:var(--tenue)">${esc(r.run)}</td></tr>`).join("") + `</tbody>`;
  $("#tabRun").querySelectorAll("tr[data-run]").forEach(tr => {
    tr.onclick = e => { if (!e.target.closest(".linkp")) { caricaRun(tr.dataset.run); window.scrollTo({top:0}); } };
    tr.onkeydown = e => { if (e.key === "Enter") caricaRun(tr.dataset.run); };
  });
  $("#tabRun").querySelectorAll(".linkp").forEach(b => b.onclick = () => apriPrompt(b.dataset.fp));
}

/* prompt e confronto fra versioni */
const dlg = $("#dlgPrompt"), selDiff = $("#selDiff");
let promptAperto = null;
function etichettaPrompt(fp){ const p = INDICE.prompt[fp] || {}; return `${p.versione || "senza versione"} (${fp})${p.prima ? ", prima run " + p.prima : ""}`; }
function apriPrompt(fp){
  if (!INDICE.prompt[fp]) { alert("Il testo di questo prompt non è nella cartella della run."); return; }
  promptAperto = fp;
  $("#dlgTitolo").textContent = "Prompt " + etichettaPrompt(fp);
  selDiff.innerHTML = `<option value="">Nessun confronto: testo completo</option>` +
    Object.keys(INDICE.prompt).filter(k => k !== fp).map(k => `<option value="${esc(k)}">${esc(etichettaPrompt(k))}</option>`).join("");
  mostraPrompt();
  dlg.showModal();
}
function righeDiff(a, b){
  const n = a.length, m = b.length, dp = Array.from({length:n+1}, () => new Uint16Array(m+1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = a[i] === b[j] ? dp[i+1][j+1] + 1 : Math.max(dp[i+1][j], dp[i][j+1]);
  const out = []; let i = 0, j = 0;
  while (i < n && j < m){
    if (a[i] === b[j]){ out.push([" ", a[i]]); i++; j++; }
    else if (dp[i+1][j] >= dp[i][j+1]) out.push(["-", a[i++]]);
    else out.push(["+", b[j++]]);
  }
  while (i < n) out.push(["-", a[i++]]); while (j < m) out.push(["+", b[j++]]);
  return out;
}
function mostraPrompt(){
  const qui = INDICE.prompt[promptAperto].testo, pre = $("#testoPrompt");
  pre.parentElement.scrollTop = 0;
  if (!selDiff.value){ pre.textContent = qui; return; }
  // il confronto va sempre dal prompt piu' vecchio al piu' recente, qualunque sia quello aperto
  const pos = fp => INDICE.runs.findIndex(r => r.fp === fp);
  const [vecchio, nuovo] = pos(selDiff.value) <= pos(promptAperto) ? [selDiff.value, promptAperto] : [promptAperto, selDiff.value];
  const d = righeDiff(INDICE.prompt[vecchio].testo.split("\n"), INDICE.prompt[nuovo].testo.split("\n"));
  const CONTESTO = 2, vicino = d.map((x, k) => x[0] !== " " ? 0 : Infinity);
  for (let k = 0; k < d.length; k++) if (d[k][0] !== " ") for (let q = Math.max(0, k - CONTESTO); q <= Math.min(d.length - 1, k + CONTESTO); q++) vicino[q] = 0;
  let html = "", saltate = 0;
  const cambi = d.filter(x => x[0] !== " ").length;
  html += `<span class="salto">${cambi ? `Cosa cambia da ${esc(etichettaPrompt(vecchio))} a ${esc(etichettaPrompt(nuovo))}: ${d.filter(x => x[0]==="+").length} righe aggiunte (verde), ${d.filter(x => x[0]==="-").length} tolte (rosa).` : "I due prompt sono identici."}</span>`;
  d.forEach((x, k) => {
    if (vicino[k] === Infinity){ saltate++; return; }
    if (saltate){ html += `<span class="salto">… ${saltate} righe uguali</span>`; saltate = 0; }
    html += x[0] === "+" ? `<span class="add">+ ${esc(x[1])}</span>` : x[0] === "-" ? `<span class="del">- ${esc(x[1])}</span>` : `  ${esc(x[1])}\n`;
  });
  if (saltate) html += `<span class="salto">… ${saltate} righe uguali</span>`;
  pre.innerHTML = html;
}
selDiff.addEventListener("change", mostraPrompt);
$("#chiudiPrompt").onclick = () => dlg.close();
$("#apriPrompt").onclick = () => D && apriPrompt(D.meta.prompt);

storico();
{
  const h = decodeURIComponent((location.hash.match(/run=([^&]+)/) || [])[1] || "");
  const iniziale = INDICE.runs.some(r => r.run === h) ? h : (INDICE.runs.length ? INDICE.runs[INDICE.runs.length - 1].run : null);
  if (iniziale) caricaRun(iniziale); else $("#titolo").textContent = "Nessuna run trovata in output.";
}
</script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
