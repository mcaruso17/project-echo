#!/usr/bin/env python3
"""
ciclo.py — orchestratore del ciclo di miglioramento del prompt.

Un round: estrazione -> punteggio -> controllo regressioni -> revisore -> ATTESA
APPROVAZIONE -> applicazione. Si ferma SEMPRE al gate: nessuna modifica al prompt entra
in produzione senza che un umano l'abbia letta.

PRINCIPI DI DISEGNO
- Non re-implementa nulla. Chiama mcex.py e confronto.py come sottoprocessi e legge
  punteggio.py: una correzione allo scorer giova a entrambi gli strumenti.
- Lo stato sta su disco, non in memoria. Ogni fase e' idempotente e riscrive state.json
  atomicamente: se il processo muore, si riparte da dove si era.
- I prompt sono immutabili. prompts/vNNN.json non viene mai riscritto; una nuova versione
  e' un file nuovo. Cosi' ogni run storico resta riproducibile.
- Il migliore non si perde mai. Su rifiuto si riparte da best_version, non dall'ultima
  versione provata: evita la deriva lenta lungo un ramo sbagliato.

IL CRITERIO D'ARRESTO e' un modello OPEN che eguaglia il benchmark. La cella commerciale
e' un soffitto di riferimento e NON puo' far scattare lo stop: altrimenti il ciclo si
fermerebbe esattamente prima di rispondere alla domanda che lo motiva (un modello
installabile al Ministero puo' fare questo lavoro?).
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import subprocess
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CICLI = ROOT / "cicli"

FASI = ["SEED", "EXTRACT", "SCORE", "GATE_REGRESSIONE", "GIUDICE", "ATTESA_APPROVAZIONE", "APPLICA"]


# --------------------------------------------------------------------------- stato
def scrivi_atomico(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Ciclo:
    def __init__(self, loop_id: str):
        self.dir = CICLI / loop_id
        self.stato_f = self.dir / "state.json"
        self.config_f = self.dir / "config.json"

    @property
    def stato(self) -> dict:
        return json.loads(self.stato_f.read_text(encoding="utf-8"))

    @property
    def config(self) -> dict:
        return json.loads(self.config_f.read_text(encoding="utf-8"))

    def salva(self, s: dict) -> None:
        scrivi_atomico(self.stato_f, s)

    def round_dir(self, n: int) -> Path:
        return self.dir / "rounds" / f"r{n:03d}"

    def prompt_path(self, versione: str) -> Path:
        return self.dir / "prompts" / f"{versione}.json"


# --------------------------------------------------------------------------- init
def cmd_init(args) -> None:
    c = Ciclo(args.loop)
    if c.stato_f.exists() and not args.force:
        sys.exit(f"{c.dir} esiste gia'. Usa --force per ricominciare (i prompt non vengono toccati).")
    (c.dir / "prompts").mkdir(parents=True, exist_ok=True)
    (c.dir / "rounds").mkdir(parents=True, exist_ok=True)
    (c.dir / "candidati").mkdir(parents=True, exist_ok=True)

    v0 = c.prompt_path("v000")
    if not v0.exists():
        out = subprocess.run([sys.executable, "mcex.py", "--print-prompt"],
                             cwd=ROOT, capture_output=True, text=True, check=True)
        v0.write_text(out.stdout, encoding="utf-8")
    fp = json.loads(v0.read_text(encoding="utf-8")).get("fingerprint")

    scrivi_atomico(c.config_f, {
        "champion": [l.strip() for l in Path(args.champion).read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.startswith("#")],
        "docs_file": args.docs,
        "modello_giudice": args.giudice,
        "max_round": args.max_round,
        "budget_usd": args.budget,
        "workers": args.workers,
        # Il soffitto commerciale misura, non decide: escluso dal criterio d'arresto.
        "celle_soffitto": [s for s in
                           [l.strip() for l in Path(args.champion).read_text(encoding="utf-8").splitlines()
                            if l.strip() and not l.startswith("#")]
                           if s.startswith(("anthropic/", "openai/", "google/gemini", "x-ai/"))],
    })
    scrivi_atomico(c.stato_f, {
        "loop_id": args.loop, "creato": dt.datetime.now().isoformat(timespec="seconds"),
        "round_corrente": 1, "fase": "SEED", "versione_corrente": "v000",
        "best_version": None, "best_solved_open": -1, "frozen_solved": [],
        "storia": [], "costo_totale": 0.0,
    })
    print(f"Ciclo '{args.loop}' inizializzato in {c.dir}")
    print(f"  prompt v000 fingerprint {fp}")
    print(f"  champion: {len(c.config['champion'])} celle "
          f"(soffitto: {', '.join(c.config['celle_soffitto']) or 'nessuno'})")
    print(f"  budget {args.budget} USD, max {args.max_round} round")
    print(f"\nProssimo passo:  python ciclo.py run --loop {args.loop}")


# --------------------------------------------------------------------------- fasi
def _estrai(c: Ciclo, n: int, s: dict) -> Path:
    cfg = c.config
    rd = c.round_dir(n); rd.mkdir(parents=True, exist_ok=True)
    marker = rd / "run_dir.txt"
    cmd = [sys.executable, "-u", "mcex.py",
           "--prompt-file", str(c.prompt_path(s["versione_corrente"])),
           "--docs-file", cfg["docs_file"],
           "--models", ",".join(cfg["champion"]),
           "--thinking", "off", "--workers", str(cfg["workers"]), "--print-run-dir"]
    if marker.exists():
        # ripresa: stessa cartella, mcex rifa' solo le celle non riuscite
        cmd += ["--resume", Path(marker.read_text().strip()).name]
        print(f"  riprendo il run {Path(marker.read_text().strip()).name}")
    print(f"  {' '.join(cmd[2:])}")
    with (rd / "mcex.log").open("w", encoding="utf-8") as log:
        pr = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
        log.write(pr.stdout)
    if pr.returncode != 0:
        sys.exit(f"mcex.py e' uscito con codice {pr.returncode}. Vedi {rd/'mcex.log'}")
    run_dir = Path(pr.stdout.strip().splitlines()[-1])
    marker.write_text(str(run_dir), encoding="utf-8")
    return run_dir


def _punteggio(c: Ciclo, n: int, run_dir: Path) -> dict:
    rd = c.round_dir(n)
    pr = subprocess.run([sys.executable, "punteggio.py", str(run_dir), "--json"],
                        cwd=ROOT, capture_output=True, text=True, check=True)
    res = json.loads(pr.stdout)
    scrivi_atomico(rd / "scores.json", res)
    subprocess.run([sys.executable, "confronto.py", str(run_dir)],
                   cwd=ROOT, capture_output=True, text=True)
    return res


def _e_soffitto(cella: str, cfg: dict) -> bool:
    return cella.split("::")[0] in cfg["celle_soffitto"]


def _tabella_variazioni(res: dict, s: dict, cfg: dict) -> tuple[str, list[str]]:
    """Vinti/persi per cella rispetto al round precedente. E' la PRIMA cosa che si legge
    nella relazione: prima della prosa del revisore, perche' una regressione va vista
    subito anche se la proposta e' convincente."""
    prec = {h["cella"]: set(h["risolti"]) for h in (s["storia"][-1]["celle"] if s["storia"] else [])} \
        if s["storia"] else {}
    righe = ["| cella | solved | vs round prec. | nuovi | persi |", "|---|---|---|---|---|"]
    regressi = []
    for cella, c in sorted(res["celle"].items(), key=lambda kv: -kv[1]["solved"]):
        ora = set(c["documenti_risolti"])
        pre = prec.get(cella, set())
        nuovi, persi = ora - pre, pre - ora
        delta = f"{len(ora)-len(pre):+d}" if prec else "—"
        tag = " *(soffitto)*" if _e_soffitto(cella, cfg) else ""
        righe.append(f"| {cella}{tag} | {c['solved']}/{c['su']} | {delta} | "
                     f"{', '.join(sorted(nuovi)) or '—'} | **{', '.join(sorted(persi)) or '—'}** |")
        regressi += [f"{cella}: {d}" for d in sorted(persi & set(s["frozen_solved"]))]
    return "\n".join(righe), regressi


def _giudice(c: Ciclo, n: int, run_dir: Path, res: dict, s: dict, variazioni: str) -> dict:
    import giudice as G
    rd = c.round_dir(n)
    # Un candidato rimasto da un tentativo precedente dello STESSO round verrebbe promosso
    # al posto della proposta di adesso, senza che nulla lo segnali. Si azzera qui.
    vecchio_cand = c.dir / "candidati" / f"v{n:03d}_candidate.json"
    if vecchio_cand.exists():
        vecchio_cand.unlink()
    prompt = json.loads(c.prompt_path(s["versione_corrente"]).read_text(encoding="utf-8"))
    dett = {cella: cc["dettaglio"] for cella, cc in res["celle"].items()}
    contesto = G.costruisci_contesto(dett, run_dir, prompt, s["storia"][-3:],
                                     [h.get("motivo_rifiuto") for h in s["storia"]
                                      if h.get("motivo_rifiuto")])
    scrivi_atomico(rd / "contesto_giudice.json", contesto)
    p = G.chiama_giudice(contesto, c.config["modello_giudice"])
    errori = G.valida_proposta(p, prompt)
    p["_validazione"] = {"ok": not errori, "errori": errori}
    scrivi_atomico(rd / "proposta.json", p)
    G.scrivi_relazione(p, errori, variazioni, rd / "proposta.md")
    if not errori and (p.get("modifiche_proposte") or []):
        nuova = f"v{n:03d}"
        scrivi_atomico(c.dir / "candidati" / f"{nuova}_candidate.json", {
            "version": nuova, "parent": s["versione_corrente"],
            "system": p["prompt_proposto"]["system"],
            "user_template": p["prompt_proposto"]["user_template"],
        })
    return p


# --------------------------------------------------------------------------- run
def cmd_run(args) -> None:
    c = Ciclo(args.loop)
    s = c.stato; cfg = c.config
    n = s["round_corrente"]

    if s["fase"] == "ATTESA_APPROVAZIONE":
        _stampa_gate(c, n); return
    if n > cfg["max_round"]:
        print(f"Raggiunto il tetto di {cfg['max_round']} round. Ciclo concluso."); return
    if s["costo_totale"] >= cfg["budget_usd"]:
        print(f"Budget esaurito (${s['costo_totale']:.2f} / ${cfg['budget_usd']}). Fermo."); return

    rd = c.round_dir(n); rd.mkdir(parents=True, exist_ok=True)
    print(f"=== Round {n} — prompt {s['versione_corrente']} ===\n")

    print("[1/4] estrazione")
    run_dir = _estrai(c, n, s)

    print("[2/4] punteggio")
    res = _punteggio(c, n, run_dir)
    aperti = {k: v for k, v in res["celle"].items() if not _e_soffitto(k, cfg)}
    best_open = max(aperti.items(), key=lambda kv: kv[1]["solved"]) if aperti else (None, {"solved": -1})
    for cella, cc in sorted(res["celle"].items(), key=lambda kv: -kv[1]["solved"]):
        tag = "  (soffitto)" if _e_soffitto(cella, cfg) else ""
        print(f"      {cc['solved']:2d}/{cc['su']}  {cella}{tag}")

    print("\n[3/4] regressioni")
    variazioni, regressi = _tabella_variazioni(res, s, cfg)
    if regressi:
        print("      REGRESSIONE su documenti gia' risolti:")
        for r in regressi:
            print(f"        {r}")
    else:
        print("      nessuna regressione rispetto ai documenti gia' risolti")

    # criterio d'arresto: SOLO una cella open
    if best_open[0] and best_open[1]["solved"] == res["n_eleggibili"]:
        print(f"\n*** {best_open[0]} ha eguagliato il benchmark "
              f"({best_open[1]['solved']}/{res['n_eleggibili']}) ***")
        print("    Da confermare con un re-run della stessa cella: i provider non sono")
        print("    bit-deterministici e un risultato che non si riproduce e' rumore.")

    print("\n[4/4] revisore")
    p = _giudice(c, n, run_dir, res, s, variazioni)
    costo = (p.get("_meta") or {}).get("costo_usd") or 0.0

    s["storia"].append({
        "round": n, "versione": s["versione_corrente"], "run_dir": run_dir.name,
        "celle": [{"cella": k, "solved": v["solved"], "su": v["su"],
                   "risolti": v["documenti_risolti"]} for k, v in res["celle"].items()],
        "best_open": best_open[0], "best_open_solved": best_open[1]["solved"],
        "regressioni": regressi,
    })
    s["frozen_solved"] = sorted(set(s["frozen_solved"]) |
                                {d for v in res["celle"].values() for d in v["documenti_risolti"]})
    if best_open[1]["solved"] > s["best_solved_open"]:
        s["best_solved_open"] = best_open[1]["solved"]
        s["best_version"] = s["versione_corrente"]
    s["costo_totale"] = round(s["costo_totale"] + costo, 4)
    s["fase"] = "ATTESA_APPROVAZIONE"
    c.salva(s)
    _aggiorna_scala(c, res, cfg)
    _stampa_gate(c, n)


def _aggiorna_scala(c: Ciclo, res: dict, cfg: dict) -> None:
    """Una riga per (round, modello): e' il deliverable che risponde alla domanda
    istituzionale, e si popola gratis a ogni round."""
    import csv
    f = c.dir / "scala_hardware.csv"
    nuovo = not f.exists()
    s = c.stato
    with f.open("a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if nuovo:
            w.writerow(["round", "versione", "cella", "ruolo", "solved", "su", "documenti_risolti"])
        for cella, cc in res["celle"].items():
            w.writerow([s["round_corrente"], s["versione_corrente"], cella,
                        "soffitto" if _e_soffitto(cella, cfg) else "candidato",
                        cc["solved"], cc["su"], "; ".join(cc["documenti_risolti"])])


def _stampa_gate(c: Ciclo, n: int) -> None:
    rd = c.round_dir(n); s = c.stato
    p = json.loads((rd / "proposta.json").read_text(encoding="utf-8"))
    val = p.get("_validazione") or {}
    print(f"\n{'='*70}\nRound {n} in attesa di approvazione")
    print(f"  miglior modello open: {s['storia'][-1]['best_open']} "
          f"({s['storia'][-1]['best_open_solved']} documenti)")
    if s["storia"][-1]["regressioni"]:
        print(f"  ATTENZIONE: {len(s['storia'][-1]['regressioni'])} regressioni")
    if not val.get("ok"):
        print(f"  PROPOSTA SCARTATA dai validatori ({len(val.get('errori', []))} violazioni)")
    print(f"\n  Leggi:     {rd/'proposta.md'}")
    print(f"  Approva:   python ciclo.py approva {n} --loop {s['loop_id']} --si")
    print(f"  Modifica:  python ciclo.py approva {n} --loop {s['loop_id']} --edit")
    print(f"  Rifiuta:   python ciclo.py approva {n} --loop {s['loop_id']} --no --motivo \"...\"")
    print("="*70)


# --------------------------------------------------------------------------- approvazione
def cmd_approva(args) -> None:
    c = Ciclo(args.loop); s = c.stato
    n = args.round
    if s["fase"] != "ATTESA_APPROVAZIONE":
        sys.exit(f"Il ciclo e' in fase {s['fase']}, non c'e' nulla da approvare.")
    rd = c.round_dir(n)
    cand = c.dir / "candidati" / f"v{n:03d}_candidate.json"
    regressioni = s["storia"][-1]["regressioni"]

    if args.no:
        s["storia"][-1]["verdetto"] = "rifiutato"
        s["storia"][-1]["motivo_rifiuto"] = args.motivo or "(nessun motivo indicato)"
        # si riparte dal MIGLIORE, non dal candidato bocciato
        s["versione_corrente"] = s["best_version"] or s["versione_corrente"]
        s["round_corrente"] = n + 1
        s["fase"] = "SEED"
        c.salva(s)
        print(f"Round {n} rifiutato. Il round {n+1} riparte da {s['versione_corrente']}.")
        return

    prop_f = rd / "proposta.json"
    prop = json.loads(prop_f.read_text(encoding="utf-8")) if prop_f.exists() else {}
    if not (prop.get("_validazione") or {}).get("ok"):
        sys.exit(f"La proposta del round {n} e' stata scartata dai validatori: non c'e' nulla "
                 f"da promuovere. Usa --no per passare al round successivo.")
    if not (prop.get("modifiche_proposte") or []):
        sys.exit(f"Il revisore del round {n} non ha proposto alcuna modifica al prompt "
                 f"(le divergenze non erano risolvibili dal prompt). Usa --no per proseguire.")
    if not cand.exists():
        sys.exit(f"Nessun file candidato per il round {n}. Usa --no per proseguire.")
    # il candidato deve discendere dal prompt attualmente in uso, non da un altro ramo
    cd = json.loads(cand.read_text(encoding="utf-8"))
    if cd.get("parent") != s["versione_corrente"]:
        sys.exit(f"Il candidato deriva da {cd.get('parent')}, ma il prompt in uso e' "
                 f"{s['versione_corrente']}: candidato incoerente, non lo promuovo.")
    if regressioni and not args.accetto_regressione:
        print(f"Il round {n} fa regredire {len(regressioni)} documenti gia' risolti:")
        for r in regressioni:
            print(f"  {r}")
        sys.exit("Per accettarlo comunque: aggiungi --accetto-regressione.")

    verdetto = "approvato"
    if args.edit:
        tmp = rd / "prompt_da_modificare.json"
        shutil.copy(cand, tmp)
        editor = os.environ.get("EDITOR", "nano")
        subprocess.run([editor, str(tmp)])
        import giudice as G
        prompt_att = json.loads(c.prompt_path(s["versione_corrente"]).read_text(encoding="utf-8"))
        d = json.loads(tmp.read_text(encoding="utf-8"))
        errori = G.valida_proposta({"modifiche_proposte": [], "analisi": [],
                                    "prompt_proposto": {"system": d["system"],
                                                        "user_template": d["user_template"]}},
                                   prompt_att)
        if errori:
            print("Le tue modifiche violano i vincoli:")
            for e in errori:
                print(f"  - {e}")
            sys.exit("Niente promosso. Correggi e riprova.")
        shutil.copy(tmp, cand)
        verdetto = "approvato_con_modifiche"

    nuova = f"v{n:03d}"
    shutil.copy(cand, c.prompt_path(nuova))
    s["storia"][-1]["verdetto"] = verdetto
    s["versione_corrente"] = nuova
    s["round_corrente"] = n + 1
    s["fase"] = "SEED"
    c.salva(s)
    print(f"Round {n} {verdetto}. Prompt {nuova} promosso.")
    print(f"Prossimo:  python ciclo.py run --loop {args.loop}")


# --------------------------------------------------------------------------- stato
def cmd_stato(args) -> None:
    c = Ciclo(args.loop); s = c.stato; cfg = c.config
    print(f"Ciclo {s['loop_id']} — fase {s['fase']}, round {s['round_corrente']}")
    print(f"  prompt corrente: {s['versione_corrente']}   migliore: {s['best_version'] or '—'} "
          f"({s['best_solved_open']} documenti open)")
    print(f"  speso: ${s['costo_totale']:.2f} / ${cfg['budget_usd']}")
    print(f"  documenti mai risolti da nessuno: {len(s['frozen_solved'])} congelati\n")
    if not s["storia"]:
        print("  nessun round ancora eseguito"); return
    print(f"  {'round':>5s} {'versione':>9s} {'best open':>10s} {'solved':>7s} {'verdetto':>22s}")
    for h in s["storia"]:
        print(f"  {h['round']:>5d} {h['versione']:>9s} "
              f"{(h['best_open'] or '—').split('::')[0][-10:]:>10s} "
              f"{h['best_open_solved']:>7d} {h.get('verdetto','in attesa'):>22s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", default="L1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init"); i.set_defaults(f=cmd_init)
    i.add_argument("--champion", default="cicli/L1/champion_round.txt")
    i.add_argument("--docs", default="documenti_eleggibili.txt")
    i.add_argument("--giudice", default="anthropic/claude-opus-5.5")
    i.add_argument("--max-round", type=int, default=8)
    i.add_argument("--budget", type=float, default=80.0)
    i.add_argument("--workers", type=int, default=4)
    i.add_argument("--force", action="store_true")

    r = sub.add_parser("run"); r.set_defaults(f=cmd_run)
    a = sub.add_parser("approva"); a.set_defaults(f=cmd_approva)
    a.add_argument("round", type=int)
    a.add_argument("--si", action="store_true")
    a.add_argument("--no", action="store_true")
    a.add_argument("--edit", action="store_true")
    a.add_argument("--motivo", type=str, default=None)
    a.add_argument("--accetto-regressione", action="store_true")
    st = sub.add_parser("stato"); st.set_defaults(f=cmd_stato)

    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
