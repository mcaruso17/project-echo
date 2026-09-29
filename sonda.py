#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sonda.py -- Sonda diagnostica dello schema di risposta di OpenRouter
==============================================================================
inviare UNA chiamata "base" (un semplice "ciao") a un modello e
salvare in root l'INTERO JSON di risposta, così da poter studiare a tavolino la
struttura dei campi su cui si basa l'estrattore principale: in particolare
`usage` (prompt/completion/reasoning token), `usage.cost` (costo reale in USD),
`provider`, `id`, `finish_reason`.

La latenza NON è presente nella risposta HTTP: va misurata lato client (delta tra
invio e ricezione). Qui la registriamo e la salviamo insieme al JSON grezzo, così
lo schema del file di sonda combacia con quello prodotto dall'estrattore.

USO:
    python sonda.py                          # modello di default
    python sonda.py --model openai/gpt-5-mini
    python sonda.py --model z-ai/glm-5.2 --reasoning on

OUTPUT:
    sonda_<modello_sanificato>_<timestamp>.json   (salvato nella root)
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import datetime as dt
from pathlib import Path

import requests

# =============================================================================
# VARIABILI GLOBALI (tutte dichiarate in apertura)
# =============================================================================

ROOT_DIR: Path = Path(__file__).resolve().parent           # cartella dello script = root
SONDA_DIR: Path = ROOT_DIR                                  # salvo il JSON grezzo direttamente in root (task 9)

# Stessa chiave "usa e getta" dell'estrattore; la variabile d'ambiente ha la precedenza.
# Chiave condivisa del gruppo di lavoro che opera in questa cartella: sta nel sorgente
# di proposito, cosi' gli script funzionano per tutti senza configurazione locale.
# La variabile d'ambiente OPENROUTER_API_KEY, se impostata, ha comunque la precedenza.
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY") or ""
OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL: str = "z-ai/glm-4.7-flash"                  # economico e verificato attivo su OpenRouter
DEFAULT_USER_MESSAGE: str = "Ciao! Rispondi con una sola parola."
REQUEST_TIMEOUT: int = 120                                 # secondi
REASONING_EFFORT_ON: str = "medium"                        # livello di thinking quando --reasoning on

# Header opzionali usati da OpenRouter per le classifiche (innocui).
SITE_URL: str = "https://www.mef.gov.it"
SITE_NAME: str = "MEF decree extraction - probe"


# =============================================================================
# FUNZIONI
# =============================================================================

def sanitize_slug(slug: str) -> str:
    """Trasforma lo slug OpenRouter (es. 'openai/gpt-5-nano') in un nome file/cartella
    sicuro sostituendo '/' con '_'."""
    return slug.replace("/", "_")


def build_body(model_slug: str, user_message: str, reasoning_on: bool) -> dict:
    """Costruisce il payload minimo per una chiamata di sonda, chiedendo esplicitamente
    a OpenRouter di includere il blocco `usage` (contiene token e costo reale)."""
    body: dict = {
        "model": model_slug,
        "messages": [{"role": "user", "content": user_message}],
        "usage": {"include": True},          # restituisce usage.cost (USD) e dettaglio token
    }
    if reasoning_on:
        body["reasoning"] = {"effort": REASONING_EFFORT_ON}
    return body


def call_probe(model_slug: str, user_message: str, reasoning_on: bool) -> dict:
    """Esegue la chiamata e restituisce un dizionario con: risposta grezza, latenza
    misurata lato client, e i timestamp di invio/ricezione."""
    if not OPENROUTER_API_KEY:
        sys.exit("ERRORE: variabile d'ambiente OPENROUTER_API_KEY non impostata.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
    }
    body = build_body(model_slug, user_message, reasoning_on)

    sent_at = dt.datetime.now()
    t0 = time.perf_counter()
    resp = requests.post(OPENROUTER_URL, headers=headers,
                         data=json.dumps(body), timeout=REQUEST_TIMEOUT)
    latency_seconds = time.perf_counter() - t0
    received_at = dt.datetime.now()

    # Provo a interpretare la risposta come JSON; se fallisce, salvo il testo grezzo.
    try:
        raw_response = resp.json()
    except ValueError:
        raw_response = {"_non_json_text": resp.text}

    return {
        "request_body": body,
        "http_status": resp.status_code,
        "sent_at": sent_at.isoformat(timespec="seconds"),
        "received_at": received_at.isoformat(timespec="seconds"),
        "latency_seconds": round(latency_seconds, 3),
        "raw_response": raw_response,
    }


def summarize(probe: dict) -> None:
    """Stampa a video una sintesi leggibile dei campi più rilevanti dello schema."""
    raw = probe.get("raw_response", {})
    usage = (raw.get("usage") or {}) if isinstance(raw, dict) else {}
    print(f"HTTP status ......... {probe['http_status']}")
    print(f"Latenza (s) ......... {probe['latency_seconds']}")
    print(f"response.id ......... {raw.get('id') if isinstance(raw, dict) else None}")
    print(f"response.provider ... {raw.get('provider') if isinstance(raw, dict) else None}")
    print(f"usage.prompt_tokens . {usage.get('prompt_tokens')}")
    print(f"usage.completion .... {usage.get('completion_tokens')}")
    print(f"usage.cost (USD) .... {usage.get('cost')}")
    print(f"chiavi di usage ..... {sorted(usage.keys())}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sonda lo schema di risposta di OpenRouter.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"slug OpenRouter (default: {DEFAULT_MODEL})")
    ap.add_argument("--message", default=DEFAULT_USER_MESSAGE, help="messaggio utente da inviare")
    ap.add_argument("--reasoning", choices=["on", "off"], default="off",
                    help="attiva il thinking per osservare i reasoning token nell'usage")
    args = ap.parse_args()

    SONDA_DIR.mkdir(parents=True, exist_ok=True)
    probe = call_probe(args.model, args.message, args.reasoning == "on")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = SONDA_DIR / f"sonda_{sanitize_slug(args.model)}_{stamp}.json"
    out_path.write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")

    summarize(probe)
    print(f"\nJSON grezzo completo salvato in:\n  {out_path}")


if __name__ == "__main__":
    main()
