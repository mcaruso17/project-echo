"""Run completo di ECHO in un solo comando: controlli, estrazione, ripresa, punteggio.

Uso (da PowerShell, nella cartella 5. LLMs):
    python run_completo.py --scegli          # confronta v000 e v003 sulle run "fisso" (gratis)
    python run_completo.py --prova           # solo controlli gratuiti (check-models + dry-run)
    python run_completo.py                   # run completo: chiede conferma prima di spendere
    python run_completo.py --si              # run completo senza chiedere conferma
    python run_completo.py --riprendi <run>  # riprende una run esistente, poi punteggio
Opzioni: --prompt <file.json>  --modelli <file.txt, default: gli 8 open>  --workers N  --max-riprese N
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
PY = sys.executable

# Gli 8 modelli open del champion (senza kimi-k3 e senza il soffitto commerciale).
# Si usano se non si passa --modelli con un file.
MODELLI_OPEN = ",".join([
    "google/gemma-4-31b-it", "google/gemma-4-26b-a4b-it", "z-ai/glm-4.7-flash",
    "qwen/qwen3.6-35b-a3b", "deepseek/deepseek-v4-flash", "minimax/minimax-m3",
    "moonshotai/kimi-k2.6", "deepseek/deepseek-v4-pro-0813",
])


def tieni_sveglio(attivo: bool) -> None:
    """Su Windows impedisce lo standby finche' lo script gira (niente permessi admin)."""
    if os.name != "nt":
        return
    import ctypes
    flags = 0x80000000 | 0x00000001 if attivo else 0x80000000
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def esegui(args: list[str], log=None) -> tuple[int, str]:
    """Lancia un comando, mostra l'output in diretta, lo copia nel log, restituisce l'ultima riga."""
    print(f"\n>>> python {' '.join(args)}\n", flush=True)
    p = subprocess.Popen([PY, "-u", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=ENV, text=True, encoding="utf-8", errors="replace")
    ultima = ""
    for riga in p.stdout:
        print(riga, end="", flush=True)
        if log:
            log.write(riga); log.flush()
        if riga.strip():
            ultima = riga.strip()
    return p.wait(), ultima


def celle_da_rifare(run: Path) -> int:
    f = run / "results.jsonl"
    if not f.exists():
        return -1
    stati = [json.loads(l).get("status") for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sum(s != "ok" for s in stati)


def precisione(run: Path) -> None:
    """Atti senza CUP (Q5): ogni CUP trovato li' e' un falso positivo."""
    import pandas as pd
    el = {l.strip() for l in open("documenti_eleggibili.txt", encoding="utf-8")
          if l.strip() and not l.startswith("#")}
    r = pd.read_csv(run / "records.csv")
    stem = r["decree_file"].str.removesuffix(".pdf")
    fp = r[~stem.isin(el) & ~stem.str.contains("delibera CIPE") & r["cup"].notna()]
    print("\n=== Precisione sugli atti senza CUP ===")
    print(fp.groupby(["model_slug", "decree_file"]).size().to_string() if len(fp)
          else "nessun CUP inventato")


def scegli() -> None:
    """Punteggio delle run 'fisso' (v000 vs v003, due ripetizioni). I path nel file sono del
    Mac: si tiene solo il nome della cartella e la si cerca in output/."""
    for riga in Path("fisso_runs.txt").read_text(encoding="utf-8").splitlines():
        if not riga.strip():
            continue
        v, r, path = riga.split(" ", 2)
        run = Path("output") / path.replace("\\", "/").rstrip("/").split("/")[-1]
        print(f"\n===== {v} {r}  ({run.name})")
        if not run.exists():
            print("  cartella non trovata in output/ (OneDrive l'ha sincronizzata?)")
            continue
        out = subprocess.run([PY, "punteggio.py", str(run)], capture_output=True, text=True,
                             env=ENV, encoding="utf-8", errors="replace").stdout
        print("\n".join(l for l in out.splitlines() if "(pieno" in l) or out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=str(Path("cicli") / "manuale" / "v003.json"))
    ap.add_argument("--modelli", default=None, metavar="FILE")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-riprese", type=int, default=2)
    ap.add_argument("--scegli", action="store_true")
    ap.add_argument("--prova", action="store_true")
    ap.add_argument("--si", action="store_true")
    ap.add_argument("--riprendi", default=None, metavar="RUN")
    a = ap.parse_args()

    if a.scegli:
        return scegli()

    for f in ("mcex.py", "confronto.py", "punteggio.py", a.prompt, *([a.modelli] if a.modelli else [])):
        if not Path(f).exists():
            sys.exit(f"Manca {f} nella cartella {ROOT}")
    modelli = Path(a.modelli).read_text(encoding="utf-8").strip() if a.modelli else MODELLI_OPEN
    print(f"Modelli: {modelli}")
    base = ["--models", modelli, "--prompt-file", a.prompt]

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = open(f"run_completo_{stamp}.log", "w", encoding="utf-8")
    print(f"Log: run_completo_{stamp}.log")

    if a.riprendi:
        run = Path(a.riprendi)
    else:
        esegui(["mcex.py", "--check-models"], log)
        esegui(["mcex.py", *base, "--dry-run"], log)
        if a.prova:
            return
        if not a.si and input("\nProcedo con il run a pagamento? [s/N] ").strip().lower() != "s":
            return print("Annullato.")
        tieni_sveglio(True)
        codice, ultima = esegui(["mcex.py", *base, "--workers", str(a.workers), "--print-run-dir"], log)
        run = Path(ultima)
        if codice != 0 or not run.is_dir():
            tieni_sveglio(False)
            sys.exit(f"\nIl run si e' interrotto (codice {codice}). Controlla il log; "
                     f"se la cartella esiste, riprendi con: python run_completo.py --riprendi <cartella>")

    tieni_sveglio(True)
    for i in range(1, a.max_riprese + 1):
        n = celle_da_rifare(run)
        if n == 0:
            break
        print(f"\n{n} celle non riuscite: ripresa {i}/{a.max_riprese}")
        esegui(["mcex.py", "--resume", str(run), *base, "--workers", str(a.workers)], log)
    tieni_sveglio(False)

    n = celle_da_rifare(run)
    if n:
        print(f"\nATTENZIONE: {n} celle ancora non riuscite dopo le riprese (vedi runs.csv).")
    esegui(["punteggio.py", str(run), "--dettaglio"], log)
    esegui(["confronto.py", str(run)], log)
    precisione(run)
    print(f"\nFatto. Run: {run}")


if __name__ == "__main__":
    main()
