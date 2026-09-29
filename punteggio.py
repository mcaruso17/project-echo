#!/usr/bin/env python3
"""
punteggio.py — exact set match per documento contro il benchmark manuale di Alessandro.

E' il CRITERIO D'ARRESTO del ciclo di miglioramento del prompt, e sta separato da
confronto.py di proposito: confronto.py misura precision/recall/F1 aggregati (segnale
continuo, utile al giudice), qui invece si risponde a una domanda binaria per ciascun
documento — "il modello ha estratto ESATTAMENTE le stesse coppie CUP-importo di
Alessandro, senza mancarne e senza aggiungerne?".

PROIEZIONE USATA (misurata, non supposta). Delle quattro combinazioni possibili di
(filtro su livello_importo) x (dividere o no l'importo aggregato per il numero di CUP),
sul run 20260921-164339__d02c735d i risultati sono:

    solo "totale", diviso      13/17 fable-5   9/17 deepseek    <-- questa
    solo "totale", non diviso   7/17            5/17
    tutti i record, diviso      9/17            7/17
    tutti i record, non diviso  4/17            4/17

Si usa quindi la prima: e' il livello "pairtot" che confronto.py gia' calcola. Prendere
tutti i record fa esplodere i falsi positivi perche' le quote annuali entrano in
concorrenza con i totali.

DUE LIVELLI DI CHIAVE, scelti documento per documento.
  - base  : (CUP, importo). Vale dove il benchmark ha solo DOCUMENTO / CUP / IMPORTO.
  - pieno : (CUP, importo, capitolo/PG, tipologia). Vale dove il benchmark ha compilato
            CAPITOLO/PG e TIPOLOGIA su TUTTE le righe del documento. E' il first best: il
            sistema deve collegare la fonte dell'impegno (PG), la sostanza (importo), il
            progetto (CUP) e la natura dell'azione (assegnato, impegnato, rimodulato,
            revocato, ...). Il modello da' la tipologia in stato_importo.
  Un documento compilato solo in parte resta al livello base, con un avviso: una chiave
  mista dentro lo stesso documento renderebbe il match insensato. Il numero di documenti
  valutati al livello pieno esce sempre accanto al punteggio: finche' e' piccolo, "solved"
  misura soprattutto il livello base.

TOLLERANZA. Il confronto sugli importi ammette +/- 1 EUR: quando il benchmark divide un
importo aggregato fra piu' CUP, gli arrotondamenti all'euro possono non ricomporre
l'aggregato, e sarebbe un fallimento non correggibile da nessun prompt.

DENOMINATORE. Solo i documenti che hanno righe nel benchmark, meno quelli in quarantena
(bloccati da problemi di estrazione, non di prompt). Gli altri documenti di input/ NON
danno punti: fanno da guardia di precisione e devono restare a zero coppie estratte.

Uso:
    python punteggio.py <run> [--gt "Output a MANO.xlsx"] [--quarantena "doc1,doc2"]
    python punteggio.py <run> --dettaglio          # elenca FN/FP documento per documento
    python punteggio.py <run> --json               # output per l'orchestratore
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

import confronto as C

ROOT = Path(__file__).resolve().parent

# Documenti esclusi dal denominatore perche' bloccati a monte del prompt.
# "2024 delibera CIPE del 19 dicembre 2024": lo scrape di Gazzetta Ufficiale su due colonne
# interlaccia il testo e fa scattare il content_filter del provider -> 0 record. Va tolto
# dalla quarantena appena l'estrazione a due colonne e' sistemata.
QUARANTENA_DEFAULT = ("2024 delibera CIPE del 19 dicembre 2024",)

TOLLERANZA_EUR = 1.0


# Una riga confrontabile: (CUP, importo, capitolo/PG, tipologia). Al livello base gli ultimi
# due campi sono None sia nel benchmark sia nella predizione, e il confronto si riduce alla
# coppia CUP-importo di sempre.
Riga = tuple[str, float, "str | None", "str | None"]


def pred_pairs(entry: dict, pieno: bool = False) -> set[Riga]:
    """Proiezione 'pairtot': solo i record di livello totale, importo diviso fra i CUP."""
    return {
        (r["cup"], r["imp_quota"],
         r["cap_pg"] if pieno else None, r["tipologia"] if pieno else None)
        for r in C.explode_records(entry)
        if r["livello"] == "totale"
    }


def _ordine(r: Riga) -> tuple:
    # None non si ordina contro str: serve una chiave esplicita per restare deterministici.
    return (r[0], r[1], r[2] or "", r[3] or "")


def insiemi_uguali(gt: set[Riga], pred: set[Riga]) -> tuple[bool, list, list]:
    """Uguaglianza di insiemi con tolleranza sull'importo.

    Non basta gt == pred: con +/- 1 EUR serve un abbinamento, perche' un importo predetto
    puo' essere compatibile con piu' di un importo del benchmark. Abbinamento greedy, ogni
    elemento consumato una sola volta; l'ordinamento rende il risultato deterministico."""
    restanti = sorted(pred, key=_ordine)
    fn = []
    for g in sorted(gt, key=_ordine):
        g_cup, g_imp, g_pg, g_tip = g
        trovato = None
        for i, (p_cup, p_imp, p_pg, p_tip) in enumerate(restanti):
            if (p_cup == g_cup and abs(p_imp - g_imp) <= TOLLERANZA_EUR
                    and p_pg == g_pg and p_tip == g_tip):
                trovato = i
                break
        if trovato is None:
            fn.append(g)
        else:
            restanti.pop(trovato)
    fp = restanti
    return (not fn and not fp), fn, fp


def valuta(run: str, gt_path: Path, quarantena: tuple[str, ...]) -> dict:
    run_dir, entries = C.load_run(run)
    gt = C.load_gt(gt_path)

    righe_per_doc: dict[str, list] = defaultdict(list)
    for _, r in gt.iterrows():
        righe_per_doc[r["doc"]].append(r)

    # Livello per documento: pieno solo se CAPITOLO/PG e TIPOLOGIA sono compilati ovunque.
    livello_doc: dict[str, str] = {}
    parziali = []
    for d, righe in righe_per_doc.items():
        compilate = [pd.notna(r["cap_pg"]) and pd.notna(r["tipologia"]) for r in righe]
        livello_doc[d] = "pieno" if all(compilate) else "base"
        if any(compilate) and not all(compilate):
            parziali.append(d)

    gt_per_doc: dict[str, set] = {
        d: {(r["cup"], r["imp"],
             r["cap_pg"] if livello_doc[d] == "pieno" else None,
             r["tipologia"] if livello_doc[d] == "pieno" else None) for r in righe}
        for d, righe in righe_per_doc.items()
    }

    con_benchmark = set(gt_per_doc)
    in_quarantena = sorted(con_benchmark & set(quarantena))

    # Un documento che il run NON ha processato non e' un fallimento: e' un'assenza. Contarlo
    # come errore gonfierebbe il denominatore e, peggio, lo manderebbe al revisore come se il
    # modello avesse sbagliato — che e' esattamente il modo di far inventare regole per
    # problemi inesistenti. Il denominatore sono i documenti con benchmark EFFETTIVAMENTE
    # eseguiti.
    eseguiti = {C.doc_key(e["decree_file"]) for e in entries}
    eleggibili = sorted(con_benchmark - set(quarantena) & eseguiti)
    non_eseguiti = sorted(con_benchmark - set(quarantena) - eseguiti)

    per_cella = defaultdict(list)
    for e in entries:
        per_cella[f"{e['model_slug']}::{e['thinking']}"].append(e)

    risultato = {
        "run": run_dir.name,
        "n_eleggibili": len(eleggibili),
        "eleggibili": eleggibili,
        "non_eseguiti": non_eseguiti,
        "quarantena": in_quarantena,
        "tolleranza_eur": TOLLERANZA_EUR,
        "livello_pieno": sorted(d for d in eleggibili if livello_doc[d] == "pieno"),
        "compilati_in_parte": sorted(parziali),
        "celle": {},
    }

    for cella, ents in sorted(per_cella.items()):
        pred_per_doc = {
            C.doc_key(e["decree_file"]):
                pred_pairs(e, livello_doc.get(C.doc_key(e["decree_file"])) == "pieno")
            for e in ents
        }
        stato_per_doc = {C.doc_key(e["decree_file"]): e.get("status") for e in ents}

        risolti, dettaglio = [], []
        for d in eleggibili:
            g = gt_per_doc[d]
            p = pred_per_doc.get(d, set())
            ok, fn, fp = insiemi_uguali(g, p)
            if ok:
                risolti.append(d)
            dettaglio.append({
                "doc": d, "livello": livello_doc[d], "solved": ok, "status": stato_per_doc.get(d, "assente"),
                "gt": len(g), "pred": len(p), "fn": len(fn), "fp": len(fp),
                "fn_righe": fn, "fp_righe": fp,
            })

        # Guardia di precisione: i documenti senza benchmark devono restare vuoti.
        guardia_violata = sorted(
            d for d, pr in pred_per_doc.items()
            if d not in con_benchmark and pr
        )

        risultato["celle"][cella] = {
            "solved": len(risolti),
            "solved_pieno": sum(1 for d in risolti if livello_doc[d] == "pieno"),
            "su": len(eleggibili),
            "documenti_risolti": risolti,
            "guardia_violata": guardia_violata,
            "dettaglio": dettaglio,
        }
    return risultato


def stampa(res: dict, dettaglio: bool) -> None:
    print(f"Run: {res['run']}")
    print(f"Denominatore: {res['n_eleggibili']} documenti eleggibili "
          f"(con benchmark, meno {len(res['quarantena'])} in quarantena)")
    for d in res["quarantena"]:
        print(f"    quarantena: {d}")
    print(f"Livello pieno (CUP + importo + capitolo/PG + tipologia): "
          f"{len(res['livello_pieno'])} documenti; gli altri al livello base (CUP + importo)")
    for d in res["compilati_in_parte"]:
        print(f"    ATTENZIONE: {d} ha CAPITOLO/PG o TIPOLOGIA solo su alcune righe: valutato al livello base")
    if res.get("non_eseguiti"):
        print(f"    ATTENZIONE: {len(res['non_eseguiti'])} documenti con benchmark NON sono stati "
              f"processati da questo run e restano fuori dal punteggio:")
        for d in res["non_eseguiti"][:6]:
            print(f"      {d}")
        if len(res["non_eseguiti"]) > 6:
            print(f"      ... e altri {len(res['non_eseguiti']) - 6}")
    print()
    for cella, c in res["celle"].items():
        avviso = ""
        if c["guardia_violata"]:
            avviso = f"   ATTENZIONE: guardia di precisione violata su {len(c['guardia_violata'])} doc"
        print(f"  {c['solved']:2d}/{c['su']}  (pieno {c['solved_pieno']}/{len(res['livello_pieno'])})"
              f"   {cella}{avviso}")
        for d in c["guardia_violata"]:
            print(f"           coppie CUP-importo su un atto senza benchmark: {d}")
    if not dettaglio:
        return
    for cella, c in res["celle"].items():
        print(f"\n=== {cella} ===")
        for r in c["dettaglio"]:
            segno = "OK " if r["solved"] else "NO "
            print(f"  {segno} {r['doc']:45s} [{r['livello']:5s}] gt={r['gt']:3d} pred={r['pred']:3d} "
                  f"fn={r['fn']:2d} fp={r['fp']:2d}  [{r['status']}]")
            for etichetta, righe in (("manca  ", r["fn_righe"]), ("in piu'", r["fp_righe"])):
                for cup, imp, pg, tip in righe:
                    extra = f"  {pg or '-':>8s}  {tip or '-'}" if r["livello"] == "pieno" else ""
                    print(f"         {etichetta} {cup:16s} {imp:>18,.0f}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="cartella-run (nome sotto output/ oppure percorso)")
    ap.add_argument("--gt", type=Path, default=C.GT_DEFAULT)
    ap.add_argument("--quarantena", type=str, default=None,
                    help="documenti da escludere dal denominatore, separati da ';'")
    ap.add_argument("--dettaglio", action="store_true", help="elenca FN/FP per documento")
    ap.add_argument("--json", action="store_true", help="output JSON per l'orchestratore")
    ap.add_argument("--csv", type=Path, default=None, help="scrive la tabella per documento")
    args = ap.parse_args()

    quar = tuple(s.strip() for s in args.quarantena.split(";") if s.strip()) \
        if args.quarantena is not None else QUARANTENA_DEFAULT

    res = valuta(args.run, args.gt, quar)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, default=list))
    else:
        stampa(res, args.dettaglio)

    if args.csv:
        righe = [{"cella": cella, **{k: v for k, v in r.items() if k not in ("fn_righe", "fp_righe")}}
                 for cella, c in res["celle"].items() for r in c["dettaglio"]]
        pd.DataFrame(righe).to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"\nTabella per documento -> {args.csv}")


if __name__ == "__main__":
    main()
