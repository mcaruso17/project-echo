#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
estrattore.py -- Estrazione dati da decreti PA italiani con LLM (OpenRouter)
======================================================================================
Benchmark costo-qualità di più LLM sull'estrazione di azioni di finanziamento
(CUP / capitolo / piano gestionale / importo / annualità) da decreti ministeriali
in PDF. Ogni modello riceve lo STESSO prompt congelato e lo STESSO testo, così i
confronti sono controllati e riproducibili (cfr. paper TRM, sezione Metodologia).

COSA FA (in sintesi):
  1. Legge in modo RICORSIVO tutti i PDF dentro ./input (anche in sottocartelle).
  2. Estrae il testo di ogni PDF (una sola volta) e salta i PDF-immagine.
  3. Per ogni cella dell'esperimento -> (documento x modello x thinking ON/OFF)
     interroga OpenRouter, misura latenza e costo, valida l'output JSON.
  4. Salva i risultati in una struttura a cascata dentro ./output:

        output/<timestamp>__<hash_prompt>/          <- una cartella per RUN
          <hash_prompt>.txt                          <- prompt completo (system+user)
          run_manifest.json                          <- griglia modelli, prezzi, parametri
          log/                                       <- solo se LOG_RAW_RESPONSES=True
            <ts>__<modello>__<doc>__think-off__att1.txt   <- scambio HTTP integrale, 1 file per tentativo
          <modello_sanificato>/                      <- '/' dello slug sostituito con '_'
            <documento>__think-off.json              <- un file per (documento x thinking)
            <documento>__think-on.json
          results.jsonl                              <- aggregato di massa: 1 riga JSON per cella
          runs.csv / records.csv / comparison.csv    <- viste comode (CSV) derivate

THINKING (condizione sperimentale, cfr. paper):
  OpenRouter, di default, IGNORA i parametri non supportati: passare `reasoning` a un
  modello non-thinking non dà errore ma non fa nulla -> falserebbe lo sweep ON. Per
  questo ogni modello dichiara `reasoning` in {none, optional, mandatory} e le celle
  valide sono: OFF se reasoning in {none, optional}; ON se reasoning in {optional,
  mandatory}. Un modello non-thinking NON compare mai nello sweep ON. I reasoning
  token sono fatturati come output token: entrano quindi nel costo.

DIPENDENZE:
    pip install requests pdfplumber pandas
    (pandas serve solo per l'export CSV; l'estrazione funziona anche senza.)

PRIMA DI LANCIARE:
    1. La chiave OpenRouter (usa e getta, a ricarica) è già impostata nel blocco CONFIG;
       in alternativa la variabile d'ambiente OPENROUTER_API_KEY ha la precedenza.
    2. Congelare gli slug con:   python estrattore.py --check-models
    3. Verificare che i decreti siano pubblici (Gazzetta Ufficiale) prima di inviarli
       a provider terzi.
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import argparse
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests

# pdfplumber viene importato in modo lazy dentro extract_pdf_text, così i comandi che
# non toccano i PDF (--check-models, --list-models) funzionano anche senza averlo installato.


# =============================================================================
# VARIABILI GLOBALI / CONFIG  (tutte dichiarate in apertura)
# =============================================================================

# --- Percorsi: SEMPRE relativi alla root = cartella che contiene lo script -----------
ROOT_DIR: Path = Path(__file__).resolve().parent
INPUT_DIR: Path = ROOT_DIR / "input"          # qui vanno i PDF (lettura ricorsiva)
OUTPUT_DIR: Path = ROOT_DIR / "output"        # qui nascono le cartelle-run

# --- Credenziali e endpoint OpenRouter ------------------------------------------------
# Chiave "usa e getta" a ricarica: può stare hardcoded (OneDrive considerato sicuro).
# Chiave condivisa del gruppo di lavoro che opera in questa cartella: sta nel sorgente
# di proposito, cosi' gli script funzionano per tutti senza configurazione locale.
# La variabile d'ambiente OPENROUTER_API_KEY, se impostata, ha comunque la precedenza.
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY") or ""
OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
MODELS_LIST_URL: str = "https://openrouter.ai/api/v1/models"

# --- Parametri di generazione ---------------------------------------------------------
TEMPERATURE: float = 0.0             # estrazione deterministica, comparabile tra modelli
MAX_OUTPUT_TOKENS: int = 8000        # alzare per decreti con molte assegnazioni
REQUEST_TIMEOUT: int = 180           # secondi per singola chiamata HTTP

# FLAG 1 — Tetto ai token di output.
#   False (default) -> il body contiene "max_tokens": MAX_OUTPUT_TOKENS (comportamento storico).
#   True            -> "max_tokens" NON viene inviato affatto: nessun tetto lato script, vale
#                      solo il massimo del modello/provider.
# ATTENZIONE (metodologica ed economica): togliere il tetto elimina i troncamenti
# (finish_reason="length") ma toglie anche l'unico freno alla spesa per chiamata. Su un modello
# con reasoning attivo un loop degenere può bruciare decine di migliaia di token di output.
# Il flag viene registrato nel run_manifest.json: run con e senza tetto NON sono confrontabili
# alla lettera sul costo.
DISABLE_MAX_TOKENS: bool = False

# FLAG 2 — Logging integrale delle risposte OpenRouter.
#   False (default) -> nessun log grezzo (comportamento storico).
#   True            -> dentro la cartella-run nasce una sottocartella "log/" con UN file .txt
#                      per OGNI tentativo HTTP (non solo per quello riuscito): contiene il body
#                      inviato, lo status, gli header di risposta e il corpo INTEGRALE così come
#                      restituito dal servizio, senza parsing né troncamenti.
# Serve per audit/riproducibilità: il JSON per-cella è già normalizzato, qui resta il grezzo.
# Costo: spazio su disco (una risposta con reasoning può pesare centinaia di KB).
LOG_RAW_RESPONSES: bool = False
LOG_DIR_NAME: str = "log"            # nome della sottocartella dei log dentro la cartella-run
LOG_INCLUDE_REQUEST: bool = True     # accoda al log anche il body inviato (la API key NON è mai scritta)

MAX_RETRIES: int = 4                 # tentativi su 429 / 5xx / errori di rete
RETRY_BACKOFF_CAP: int = 30          # tetto (secondi) del backoff esponenziale
RETRY_STATUS: tuple = (429, 500, 502, 503, 504)
USE_JSON_RESPONSE_FORMAT: bool = True  # chiede output JSON dove supportato (con fallback su 400)
REASONING_EFFORT_ON: str = "medium"    # livello di thinking usato nello sweep ON

# THINKING: modalità di DEFAULT dell'esperimento (sovrascrivibile dall'argomento --thinking)
# Governa quali "celle" (documento x modello x thinking) vengono eseguite:
#   "off"  -> ogni modello gira con reasoning DISATTIVATO (reasoning:{enabled:false});
#   "on"   -> girano SOLO i modelli reasoning-capable, con reasoning ATTIVO (effort=REASONING_EFFORT_ON);
#   "both" -> due celle separate per modello (OFF e ON), per confrontare qualità vs costo aggiuntivo.
# Nota di rigore: un modello non-thinking non entra mai nello sweep ON, e uno a reasoning
# "mandatory" non entra mai nello sweep OFF -> le celle restano sempre "pulite".
THINKING_DEFAULT: str = "off"

# --- Rilevamento PDF-immagine ---------------------------------------------------------
# Se un PDF rende meno di questa soglia di caratteri per pagina, lo consideriamo una
# scansione (serve OCR o vision) e lo saltiamo per non spendere su input vuoto.
MIN_CHARS_PER_PAGE: int = 80

# --- Selezione dei documenti (--doc) --------------------------------------------------
# Il pattern di --doc è per default un AND di token sul percorso relativo a input/
# ('270 2023' seleziona 'DM 270-2023.pdf'). Con questo prefisso si passa alla regex vera.
DOC_REGEX_PREFIX: str = "re:"

# --- Header opzionali per le classifiche OpenRouter (innocui) -------------------------
SITE_URL: str = "https://www.mef.gov.it"
SITE_NAME: str = "MEF decree extraction"

# --- Costo: metodologia -------------------------------------------------------------
# Il costo AUTOREVOLE è quello REALE addebitato da OpenRouter, restituito in `usage.cost`
# (salvato come `cost_usd_api`): è SEMPRE aggiornato, chiamata per chiamata. I prezzi di
# listino `price_in`/`price_out` nella griglia servono solo alla RICOSTRUZIONE del costo dai
# token (`cost_usd_reconstructed`), come controprova: sono congelati a PRICE_LIST_DATE e
# possono divergere se OpenRouter aggiorna i prezzi.
PRICE_LIST_DATE: str = "2026-07-23"    # data di congelamento dei $/Mtoken di listino (solo per la ricostruzione)

# --- Espressioni regolari riusabili ---------------------------------------------------
HEADER_NOISE_RE: re.Pattern = re.compile(
    r"gazzetta ufficiale|serie generale|^\s*pag\.?\s*\d+\s*$", re.IGNORECASE)
CUP_RE: re.Pattern = re.compile(r"^[A-Z0-9]{15}$")   # il CUP è un codice alfanumerico di 15 caratteri
MULTISPACE_RE: re.Pattern = re.compile(r"[ \t]{2,}")

# --- Segnaposto del testo del decreto nel template utente -----------------------------
DECREE_PLACEHOLDER: str = "<<DECREE_TEXT>>"


# =============================================================================
# GRIGLIA DEI MODELLI
# =============================================================================
# Metadati per ogni modello: la griglia è a 4 fasce (small/medium/frontier/flagship) x famiglie.
#  - kind: "commercial" (pesi NON pubblici -> la fascia è il posizionamento/prezzo) oppure
#          "open" (pesi pubblici -> riportiamo i parametri totali e attivi per i MoE).
#  - reasoning: "none" (no thinking) | "optional" (attivabile) | "mandatory" (sempre attivo).
#    -> è QUESTO campo a separare i modelli thinking dai non-thinking nelle celle ON/OFF.
#  - price_in / price_out: $/Mtoken di listino (data PRICE_LIST_DATE) usati SOLO per la
#    ricostruzione del costo (cost_usd_reconstructed). Il costo AUTOREVOLE è invece
#    cost_usd_api (usage.cost di OpenRouter), sempre aggiornato: questi listini possono
#    invecchiare, il costo reale addebitato no.
# NOTA: gli slug vanno congelati con --check-models prima dell'esperimento definitivo.

@dataclass(frozen=True)
class ModelSpec:
    slug: str                         # slug OpenRouter, es. "moonshotai/kimi-k2.6"
    family: str                       # "anthropic","openai","google","zai","moonshot","deepseek","qwen"
    kind: str                         # "commercial" | "open"
    tier: str                         # "small" | "medium" | "frontier" | "flagship"
    vision: bool                      # supporta input immagine (per i PDF-immagine, uso futuro)
    reasoning: str                    # "none" | "optional" | "mandatory"
    price_in: Optional[float]         # $/Mtoken input di listino (None se ignoto)
    price_out: Optional[float]        # $/Mtoken output di listino (None se ignoto)
    params_total: Optional[str] = None    # solo open, es. "1T"  (None per i commerciali)
    params_active: Optional[str] = None   # solo open MoE, es. "32B"

    @property
    def dir_name(self) -> str:
        """Nome cartella di output: slug con '/' sostituito da '_' (cfr. task 8)."""
        return self.slug.replace("/", "_")


# Firma pronta all'uso per aggiungere una riga (ordine esatto dei campi di ModelSpec):
#   ModelSpec(slug, family, kind, tier, vision, reasoning, price_in, price_out, params_total=None, params_active=None)
MODELS: list[ModelSpec] = [
    # ---- COMMERCIALI: Anthropic ------------------------------------------------------
    ModelSpec("anthropic/claude-haiku-4.5", "anthropic", "commercial", "small",    True, "optional", 1.0, 5.0),
    ModelSpec("anthropic/claude-sonnet-5",  "anthropic", "commercial", "medium",   True, "optional", 2.0, 10.0),
    ModelSpec("anthropic/claude-opus-4.8",  "anthropic", "commercial", "frontier", True, "optional", 5.0, 25.0),
    ModelSpec("anthropic/claude-fable-5",   "anthropic", "commercial", "flagship", True, "optional", 10.0, 50.0),
    # ---- COMMERCIALI: OpenAI (generazione 5.6; flagship = gpt-5.5-pro) ----------------
    ModelSpec("openai/gpt-5.6-luna",  "openai", "commercial", "small",    True, "optional", 1.0, 6.0),
    ModelSpec("openai/gpt-5.6-terra", "openai", "commercial", "medium",   True, "optional", 2.5, 15.0),
    ModelSpec("openai/gpt-5.6-sol",   "openai", "commercial", "frontier", True, "optional", 5.0, 30.0),
    ModelSpec("openai/gpt-5.5-pro",   "openai", "commercial", "flagship", True, "optional", 30.0, 180.0),
    # ---- COMMERCIALI: Google (Gemini) ------------------------------------------------
    ModelSpec("google/gemini-3.1-flash-lite",  "google", "commercial", "small",    True, "optional", 0.25, 1.5),
    ModelSpec("google/gemini-3.6-flash",       "google", "commercial", "medium",   True, "optional", 1.5, 7.5),
    ModelSpec("google/gemini-3.1-pro-preview", "google", "commercial", "frontier", True, "optional", 2.0, 12.0),
    # ---- OPEN: Z.AI (GLM) — verticale completa ---------------------------------------
    ModelSpec("z-ai/glm-4.7-flash", "zai", "open", "small",    False, "optional", 0.06, 0.40, params_total="~30B"),
    ModelSpec("z-ai/glm-5",         "zai", "open", "medium",   False, "optional", 0.60, 1.92),
    ModelSpec("z-ai/glm-5.2",       "zai", "open", "frontier", False, "optional", 0.77, 2.42, params_total="744B"),
    # ---- OPEN: Moonshot (Kimi) — nasce grande: solo medium/frontier (nessuno small a catalogo)
    ModelSpec("moonshotai/kimi-k2.6", "moonshot", "open", "medium",   True, "optional", 0.66, 3.41, params_total="~1T", params_active="32B"),
    ModelSpec("moonshotai/kimi-k3",   "moonshot", "open", "frontier", True, "optional", 3.0, 15.0, params_total="2.8T"),
    # ---- OPEN: DeepSeek — niente small pulito a catalogo (solo medium/frontier) -------
    ModelSpec("deepseek/deepseek-v4-flash", "deepseek", "open", "medium",   False, "optional", 0.09, 0.18, params_total="284B", params_active="13B"),
    # Slug DATATO (build 0813, GA del 12-08-2026) e non l'alias flottante "deepseek-v4-pro":
    # l'alias ripunta da solo alle build successive e renderebbe non confrontabili due run
    # a distanza di settimane, pur riportando lo stesso model_slug nel manifest.
    ModelSpec("deepseek/deepseek-v4-pro-0813", "deepseek", "open", "frontier", False, "optional", 0.435, 0.87, params_total="1.6T", params_active="49B"),
    # ---- OPEN: Qwen — fascia open-small (MoE compatto single-GPU, vision) ------------
    ModelSpec("qwen/qwen3.6-35b-a3b", "qwen", "open", "small", True, "optional", 0.13, 1.0, params_total="35B", params_active="3B"),
]


# =============================================================================
# PROMPT CONGELATO  (system + template utente; identico per tutti i modelli)
# =============================================================================

SYSTEM_PROMPT: str = r"""
Sei un analista esperto del bilancio dello Stato italiano e dei decreti ministeriali
(decreti ministeriali / interministeriali). Il tuo compito è estrarre, dal testo del
decreto fornito, ogni azione di finanziamento che collega linee di bilancio a progetti.

Concetti chiave:
- "capitolo di spesa" (capitolo): il capitolo di bilancio su cui lo Stato stanzia risorse.
- "piano gestionale" (piano_gestionale): una sotto-articolazione del capitolo.
- "CUP" (Codice Unico di Progetto): codice alfanumerico di 15 caratteri che identifica un
  progetto specifico (es. "J51B21000000001"). Compare SOLO quando un decreto assegna
  risorse a un progetto identificato. Molti decreti NON hanno alcun CUP: è normale.
- Un'azione di finanziamento muove denaro: un'"assegnazione/finanziamento" (positiva), un
  "definanziamento/riduzione" (negativa), o una "rimodulazione" (fondi spostati, es. da un
  piano gestionale verso un CUP; può avere una gamba positiva e una negativa).

IDENTIFICAZIONE DEL DECRETO CORRENTE

Il documento può citare numerosi decreti precedenti nel preambolo
("VISTO", "RICHIAMATO", "CONSIDERATO", ecc.).

Il decreto corrente è esclusivamente quello rappresentato dal documento
che stai analizzando, cioè quello che contiene la parte dispositiva
("DECRETA", "ART.", ecc.).

I campi:

- numero_decreto
- data_decreto
- ministero_1
- ministero_2

devono riferirsi SEMPRE al decreto corrente e MAI ai decreti citati nel
preambolo o nelle tabelle.

REGOLE DI ESTRAZIONE:
1. Estrai SOLO ciò che è esplicitamente scritto nel testo. NON dedurre, completare o
   inventare codici o importi. Se un campo non è presente, usa null.
2. Se il decreto non contiene alcuna azione di finanziamento con il dettaglio richiesto,
   restituisci un array "record" vuoto. Non restituire nulla è meglio che tirare a indovinare.
3. Importi: il formato italiano usa "." per le migliaia e "," per i decimali
   ("1.500.000,00" = un milione e cinquecentomila). Restituisci "importo_eur" come numero
   semplice (punto decimale, senza separatori) e conserva la stringa originale in
   "importo_originale". Usa un numero NEGATIVO per definanziamenti/riduzioni e per la gamba
   negativa di una rimodulazione; positivo per le assegnazioni e per la gamba positiva.
4. Un record per ogni azione di finanziamento. Se una rimodulazione toglie fondi da un
   piano gestionale e li assegna a un CUP, e sono indicate sia la riduzione all'origine sia
   l'aumento a destinazione, crea record separati per ciascuna gamba.
5. Compila "riferimento_fonte" con il punto del decreto in cui hai trovato il dato (es.
   "art. 2", "Allegato 1, riga 5"), così un umano può verificare in fretta.
6. Riporta i campi a livello di decreto (numero, data, ministeri) una sola volta al livello
   superiore, non dentro ogni record.
7. Il preambolo ("VISTO", "CONSIDERATO", "RITENUTO", ecc.) ha funzione esclusivamente descrittiva
e serve come contesto storico. NON estrarre azioni di finanziamento dal preambolo, 
salvo che il decreto rinvii esplicitamente ad un allegato contenente le nuove assegnazioni.
8. Le azioni di finanziamento devono essere estratte dalla parte
dispositiva del decreto ("DECRETA", "ART.", ecc.) e dagli allegati
richiamati dagli articoli.
9. Se una tabella contiene colonne come "Decreto di assegnazione",
"Decreto originario", "Decreto di finanziamento" o analoghe, tali valori
identificano il provvedimento con cui il progetto era stato finanziato in
precedenza e NON il decreto corrente.
10. Se il decreto corrente assegna contributi aggiuntivi a interventi già
finanziati da precedenti decreti, estrai esclusivamente il nuovo importo
assegnato dal decreto corrente. I decreti precedenti possono essere
citati eventualmente nel campo "note", ma NON devono essere utilizzati
come valore di "numero_decreto".
11. Prima di produrre il JSON esegui mentalmente questa procedura:

a) identifica il decreto corrente;
b) individua dove inizia la parte dispositiva ("DECRETA");
c) individua le nuove azioni di finanziamento;
d) verifica che "numero_decreto" appartenga al decreto corrente e non a
un decreto citato.

Restituisci UN SOLO oggetto JSON, e nient'altro (niente prosa, niente fence markdown), con
questa forma esatta:

{
  "decreto_corrente_identificato": true
  "numero_decreto": string | null,       // es. "125" o "125/2024" come stampato
  "data_decreto": string | null,         // ISO "AAAA-MM-GG"
  "ministero_1": string | null,          // ministero primo / capofila
  "ministero_2": string | null,          // secondo ministero se interministeriale, altrimenti null
  "record": [
    {
      "cup": string | null,               // codice di 15 caratteri, maiuscolo, senza spazi
      "capitolo": string | null,          // capitolo, come stampato (es. "7003")
      "piano_gestionale": string | null,  // piano gestionale, come stampato (es. "1")
      "importo_eur": number | null,       // con segno, punto decimale
      "importo_originale": string | null, // stringa originale, es. "€ 1.500.000,00"
      "tipo_azione": "assegnazione" | "definanziamento" | "rimodulazione" | "altro",
      "riferimento_fonte": string | null,
      "note": string | null
    },
    {
      "cup": string | null,               // codice di 15 caratteri, maiuscolo, senza spazi
      "capitolo": string | null,          // capitolo, come stampato (es. "7003")
       ...
    }

  ],
  "note_estrazione": string | null       // ambiguità, se il testo sembrava incompleto
}
""".strip()

USER_PROMPT_TEMPLATE: str = (
    "Analizza il seguente decreto ministeriale.\n\n"
    "Identifica innanzitutto il decreto corrente rappresentato dal documento.\n"
    "Ignora come possibili risultati i decreti citati nel preambolo o nelle "
    "tabelle (ad esempio colonne 'Decreto di assegnazione', 'Decreto originario', "
    "'Decreto di finanziamento'), salvo che il decreto corrente disponga "
    "esplicitamente una modifica o una revoca di tali decreti.\n\n"
    "Estrai esclusivamente le azioni di finanziamento disposte dal decreto corrente.\n"
    "Restituisci solo il JSON descritto nel system prompt.\n\n"
    "=== INIZIO TESTO DECRETO ===\n"
    f"{DECREE_PLACEHOLDER}\n"
    "=== FINE TESTO DECRETO ==="
)


def build_messages(decree_text: str) -> list[dict]:
    """Costruisce i messaggi chat. Il system prompt è statico (cache-abile per modello);
    il template utente contiene un segnaposto sostituito col testo del decreto — usiamo
    replace() e non format() perché il testo può contenere parentesi graffe."""
    user_content = USER_PROMPT_TEMPLATE.replace(DECREE_PLACEHOLDER, decree_text)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def prompt_fingerprint() -> str:
    """Hash MD5 corto (8 char) della parte INVARIANTE del prompt (system + template
    utente, senza il testo del decreto). Serve a sapere a posteriori 'cosa abbiamo
    lanciato' e a nominare sia la cartella-run sia il file .txt del prompt (task 7)."""
    payload = SYSTEM_PROMPT + "\n---\n" + USER_PROMPT_TEMPLATE
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def dump_prompt_file(run_dir: Path, fingerprint: str) -> None:
    """Scrive nel run il file <hash>.txt con l'intero prompt (system + template utente).
    Crea la cartella-run se non esiste ancora (è la prima scrittura del run)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"# Prompt congelato — fingerprint MD5: {fingerprint}\n"
        f"# Generato il {dt.datetime.now().isoformat(timespec='seconds')}\n\n"
        "===================== SYSTEM PROMPT =====================\n"
        f"{SYSTEM_PROMPT}\n\n"
        "================ USER PROMPT (template) =================\n"
        f"{USER_PROMPT_TEMPLATE}\n"
    )
    (run_dir / f"{fingerprint}.txt").write_text(content, encoding="utf-8")


# =============================================================================
# ESTRAZIONE TESTO DAI PDF
# =============================================================================

def clean_text(text: str) -> str:
    """Pulizia leggera e sicura: normalizza gli spazi e rimuove il rumore ripetuto di
    intestazioni/piè di pagina. NON tocca il preambolo giuridico, perché a volte gli
    importi compaiono lì."""
    lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or HEADER_NOISE_RE.search(s):
            continue
        lines.append(s)
    return MULTISPACE_RE.sub(" ", "\n".join(lines))


def extract_pdf_text(path: Path) -> tuple[str, int, bool]:
    """Restituisce (testo, n_pagine, sembra_scansione). Usa pdfplumber e accoda le
    tabelle come righe 'cella | cella', perché le mappe CUP<->importo stanno spesso
    negli allegati tabellari."""
    import pdfplumber  # import lazy

    parts: list[str] = []
    n_pages = 0
    with pdfplumber.open(str(path)) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            parts.append(page.extract_text() or "")
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for t_idx, table in enumerate(tables or []):
                parts.append(f"\n[TABELLA pag.{i} n.{t_idx + 1}]")
                for row in table:
                    cells = [(c or "").strip().replace("\n", " ") for c in row]
                    parts.append(" | ".join(cells))

    text = clean_text("\n".join(parts))
    avg_chars = (len(text) / n_pages) if n_pages else 0
    looks_scanned = avg_chars < MIN_CHARS_PER_PAGE
    return text, n_pages, looks_scanned


def find_input_pdfs(limit: Optional[int] = None) -> list[Path]:
    """Elenca in modo RICORSIVO tutti i PDF sotto INPUT_DIR (anche nelle sottocartelle),
    ordinati per percorso. Un eventuale 'limit' serve per run di prova economiche."""
    pdfs = sorted(p for p in INPUT_DIR.rglob("*.pdf") if p.is_file())
    return pdfs[:limit] if limit else pdfs


def match_pdfs(pattern: str, pdfs: Optional[list[Path]] = None) -> list[Path]:
    """Seleziona i PDF il cui PERCORSO RELATIVO a INPUT_DIR soddisfa `pattern`.

    Due modalità:
      - DEFAULT (token AND, case-insensitive): il pattern viene spezzato su spazi/virgole e
        un file è selezionato se contiene TUTTI i token. Così '270 2023' prende
        'DM 270-2023.pdf', 'dm_270_2023_all1.pdf' e 'archivio/2023/dm-270.pdf', senza
        doversi ricordare separatori, maiuscole o sottocartelle.
      - REGEX (prefisso 're:'): il resto è trattato come espressione regolare vera
        (re.search, IGNORECASE). Es.  --doc "re:dm[ _-]?270[ _-]?2023".

    Perché i token e non la regex come default: nei nomi dei decreti i separatori sono
    imprevedibili ('-', '_', ' ', '.') e caratteri come '.' o '(' hanno un significato in
    regex — un pattern ingenuo o matcha troppo o non matcha nulla. Il token AND non ha
    caratteri speciali. La regex resta disponibile per i casi che la richiedono davvero.
    """
    pool = pdfs if pdfs is not None else find_input_pdfs()
    pattern = (pattern or "").strip()
    if not pattern:
        return pool

    def rel(p: Path) -> str:
        try:
            return p.relative_to(INPUT_DIR).as_posix()
        except ValueError:
            return p.name

    if pattern.lower().startswith(DOC_REGEX_PREFIX):
        expr = pattern[len(DOC_REGEX_PREFIX):].strip()
        try:
            rx = re.compile(expr, re.IGNORECASE)
        except re.error as e:
            print(f"Regex non valida ({expr}): {e}")
            return []
        return [p for p in pool if rx.search(rel(p))]

    tokens = [t for t in re.split(r"[\s,]+", pattern.lower()) if t]
    return [p for p in pool if all(t in rel(p).lower() for t in tokens)]


def select_documents(doc_arg: Optional[str], limit: Optional[int]) -> list[Path]:
    """Documenti da processare in un run: prima il filtro --doc, POI --limit (l'ordine conta:
    limitare prima renderebbe il filtro cieco sul resto della cartella). Stampa sempre cosa ha
    selezionato: su un run a pagamento, sapere su cosa stai spendendo viene prima di tutto."""
    pool = find_input_pdfs()
    if not pool:
        return []
    if doc_arg:
        sel = match_pdfs(doc_arg, pool)
        if not sel:
            print(f"Nessun PDF in {INPUT_DIR} corrisponde a --doc '{doc_arg}'.")
            print("Suggerimento: token separati da spazio (AND), es. --doc \"270 2023\"; "
                  "oppure regex col prefisso 're:'.")
            return []
        print(f"--doc '{doc_arg}' -> {len(sel)} documento/i selezionato/i:")
        for p in sel[:20]:
            print(f"    {p.relative_to(INPUT_DIR).as_posix()}")
        if len(sel) > 20:
            print(f"    ... e altri {len(sel) - 20}")
        pool = sel
    return pool[:limit] if limit else pool


# =============================================================================
# CHIAMATA A OPENROUTER  (costo + latenza + gestione robusta dei parametri)
# =============================================================================

def build_request_body(spec: ModelSpec, messages: list[dict], thinking_on: bool) -> dict:
    """Costruisce il payload per un modello e una modalità di thinking.
    La logica di `reasoning` riflette la semantica di OpenRouter (parametri non
    supportati ignorati): per un modello non-thinking NON inviamo affatto `reasoning`."""
    body: dict = {
        "model": spec.slug,
        "messages": messages,
        "temperature": TEMPERATURE,
        "usage": {"include": True},   # restituisce usage.cost (USD reale) e dettaglio token
    }
    # FLAG 1: con DISABLE_MAX_TOKENS=True la chiave 'max_tokens' non entra proprio nel body
    # (diverso da mandare null/0, che alcuni provider rifiutano con 400).
    if not DISABLE_MAX_TOKENS:
        body["max_tokens"] = MAX_OUTPUT_TOKENS
    if USE_JSON_RESPONSE_FORMAT:
        body["response_format"] = {"type": "json_object"}
    if spec.reasoning == "none":
        pass                                            # modello non-thinking: nessun parametro reasoning
    elif thinking_on:
        body["reasoning"] = {"effort": REASONING_EFFORT_ON}   # effort implica reasoning attivo
    else:
        body["reasoning"] = {"enabled": False}          # OFF esplicito e uniforme sui modelli 'optional'
    return body


def _extract_usage(raw: dict) -> dict:
    """Estrae in modo difensivo i contatori di token e il costo dal blocco `usage`
    (i nomi dei sotto-campi possono mancare a seconda del provider)."""
    usage = raw.get("usage") or {}
    comp_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": comp_details.get("reasoning_tokens"),   # fatturati come output
        "cached_tokens": prompt_details.get("cached_tokens"),
        "cost_usd_api": usage.get("cost"),          # COSTO AUTOREVOLE: reale OpenRouter, sempre aggiornato
    }


def reconstruct_cost(spec: ModelSpec, prompt_tokens, completion_tokens) -> Optional[float]:
    """Costo RICOSTRUITO dai token e dal listino congelato (controprova secondaria: il costo
    AUTOREVOLE resta `cost_usd_api` = usage.cost di OpenRouter, sempre aggiornato).
    I reasoning token sono già inclusi nei completion_tokens. Restituisce None se manca
    un prezzo (es. listino non disponibile) o un contatore."""
    if None in (spec.price_in, spec.price_out, prompt_tokens, completion_tokens):
        return None
    return (prompt_tokens * spec.price_in + completion_tokens * spec.price_out) / 1_000_000


# --- FLAG 2: logging integrale delle risposte -----------------------------------------

def _log_file_name(spec: ModelSpec, doc_stem: str, thinking_on: bool, attempt: int) -> str:
    """Nome del file di log: ordinabile per timestamp e univoco per (modello x documento x
    thinking x tentativo). Il millisecondo evita collisioni tra retry ravvicinati."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    suffix = "think-on" if thinking_on else "think-off"
    safe_doc = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_stem or "doc")[:80]
    return f"{stamp}__{spec.dir_name}__{safe_doc}__{suffix}__att{attempt}.txt"


def log_raw_exchange(log_dir: Optional[Path], spec: ModelSpec, doc_stem: str, thinking_on: bool,
                     attempt: int, request_body: dict, resp=None,
                     network_error: Optional[str] = None,
                     latency_seconds: Optional[float] = None) -> None:
    """Scrive in <run>/log/ un .txt con lo scambio HTTP INTEGRALE di UN tentativo:
    body inviato (opzionale, senza credenziali), status, header di risposta e corpo della
    risposta esattamente come arrivato (nessun parsing, nessun troncamento).
    Vale per TUTTI i tentativi: 400/429/5xx ed errori di rete inclusi — sono quelli che
    servono davvero quando un modello si comporta male.
    Non solleva mai eccezioni: un log che fallisce non deve far saltare una chiamata pagata."""
    if not LOG_RAW_RESPONSES or log_dir is None:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        parts: list[str] = [
            "=" * 78,
            f"TIMESTAMP      : {dt.datetime.now().isoformat(timespec='milliseconds')}",
            f"MODELLO        : {spec.slug}   (tier {spec.tier}, reasoning {spec.reasoning})",
            f"THINKING       : {'on' if thinking_on else 'off'}",
            f"DOCUMENTO      : {doc_stem}",
            f"TENTATIVO      : {attempt}/{MAX_RETRIES}",
            f"ENDPOINT       : {OPENROUTER_URL}",
            f"LATENZA (s)    : {round(latency_seconds, 3) if latency_seconds is not None else 'n/d'}",
            "=" * 78,
            "",
        ]
        if LOG_INCLUDE_REQUEST:
            parts += ["----- REQUEST BODY (credenziali NON incluse) -----",
                      json.dumps(request_body, ensure_ascii=False, indent=2), ""]
        if network_error is not None:
            parts += ["----- ERRORE DI RETE (nessuna risposta HTTP) -----", network_error, ""]
        if resp is not None:
            parts += [
                "----- RESPONSE STATUS -----",
                f"{resp.status_code} {getattr(resp, 'reason', '')}".strip(),
                "",
                "----- RESPONSE HEADERS -----",
                "\n".join(f"{k}: {v}" for k, v in dict(resp.headers).items()),
                "",
                "----- RESPONSE BODY (integrale, non elaborato) -----",
                resp.text,
                "",
            ]
        path = log_dir / _log_file_name(spec, doc_stem, thinking_on, attempt)
        path.write_text("\n".join(parts), encoding="utf-8")
    except Exception as e:          # il logging non è mai bloccante
        print(f"    (log grezzo non scritto: {e})")


def call_openrouter(spec: ModelSpec, messages: list[dict], thinking_on: bool,
                    log_dir: Optional[Path] = None, doc_stem: str = "doc") -> dict:
    """Chiama OpenRouter con retry su 429/5xx/rete e con fallback sui parametri opzionali
    in caso di 400 (alcuni modelli non accettano response_format/reasoning). Misura la
    latenza lato client. Restituisce un dizionario normalizzato; solleva RuntimeError su
    fallimento irrecuperabile."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Nessuna API key: impostare la variabile d'ambiente OPENROUTER_API_KEY.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
    }
    body = build_request_body(spec, messages, thinking_on)

    # Strategie di degradazione dei parametri opzionali, provate in ordine su HTTP 400.
    stripped: list[str] = []
    last_err: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        sent_at = dt.datetime.now()
        t0 = time.perf_counter()
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 data=json.dumps(body), timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last_err = str(e)
            log_raw_exchange(log_dir, spec, doc_stem, thinking_on, attempt, body,
                             network_error=f"{type(e).__name__}: {e}",
                             latency_seconds=time.perf_counter() - t0)
            _sleep_backoff(attempt, f"errore di rete ({e})")
            continue
        latency_seconds = time.perf_counter() - t0
        received_at = dt.datetime.now()

        # Log INTEGRALE del tentativo: unico punto, così finisce a disco anche ciò che poi
        # viene scartato (400 con parametro rimosso, 429, 5xx) e non solo la risposta buona.
        log_raw_exchange(log_dir, spec, doc_stem, thinking_on, attempt, body,
                         resp=resp, latency_seconds=latency_seconds)

        # 400: provo a togliere un parametro opzionale alla volta e riprovo subito.
        if resp.status_code == 400 and ("response_format" in body or "reasoning" in body):
            if "response_format" in body:
                body.pop("response_format"); stripped.append("response_format")
            elif "reasoning" in body:
                body.pop("reasoning"); stripped.append("reasoning")
            last_err = f"HTTP 400: {resp.text[:200]}"
            print(f"    HTTP 400 -> rimuovo '{stripped[-1]}' e riprovo")
            continue

        if resp.status_code in RETRY_STATUS:
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            _sleep_backoff(attempt, f"HTTP {resp.status_code}")
            continue

        resp.raise_for_status()
        raw = resp.json()
        choice = (raw.get("choices") or [{}])[0]
        usage = _extract_usage(raw)
        return {
            "content": (choice.get("message") or {}).get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "sent_at": sent_at.isoformat(timespec="seconds"),
            "received_at": received_at.isoformat(timespec="seconds"),
            "latency_seconds": round(latency_seconds, 3),
            "stripped_params": stripped,
            "response_meta": {
                "id": raw.get("id"),
                "provider": raw.get("provider"),
                "model": raw.get("model"),
            },
            **usage,
        }

    raise RuntimeError(f"Chiamata OpenRouter fallita dopo {MAX_RETRIES} tentativi: {last_err}")


def _sleep_backoff(attempt: int, reason: str) -> None:
    """Backoff esponenziale con tetto, con messaggio diagnostico."""
    wait = min(2 ** attempt, RETRY_BACKOFF_CAP)
    print(f"    {reason}, retry {attempt}/{MAX_RETRIES} tra {wait}s")
    time.sleep(wait)


# =============================================================================
# PARSING & NORMALIZZAZIONE DELL'OUTPUT DEL MODELLO
# =============================================================================

def extract_json_block(text: str) -> Optional[dict]:
    """Restituisce il primo oggetto JSON valido di primo livello dalla risposta del
    modello, tollerando i fence markdown e testo spurio attorno."""
    if not text:
        return None
    t = re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", text.strip()).strip()).strip()
    try:
        return json.loads(t)                       # percorso veloce
    except Exception:
        pass
    start = t.find("{")                            # fallback: bilanciamento delle graffe
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    return None
    return None


def repair_truncated_json(t: str) -> Optional[str]:
    """Tenta di chiudere un oggetto JSON interrotto a metà: scandisce il testo tenendo conto
    delle stringhe e degli escape, si ferma all'ultimo elemento COMPLETO e aggiunge i
    caratteri di chiusura mancanti. Restituisce None se non c'è nulla da riparare.

    Serve perché un provider può restituire un corpo incompleto dichiarando finish_reason
    'stop': in quel caso manca solo la graffa finale e senza riparazione si perdono decine
    di record già pagati. La riparazione è SEMPRE tracciata (json_repaired=True): un modello
    che emette JSON non valido resta un dato dell'esperimento, non un dettaglio da nascondere."""
    start = t.find("{")
    if start == -1:
        return None
    stack: list[str] = []
    in_str = False
    esc = False
    last_safe = None          # offset dopo l'ultimo elemento chiuso correttamente
    for i in range(start, len(t)):
        ch = t[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            if not in_str:
                last_safe = i + 1
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                last_safe = i + 1
            else:
                return None                  # parentesi sbilanciate: non è un troncamento
        elif ch == ",":
            last_safe = i                    # taglio PRIMA della virgola: niente trailing comma
    if not stack:
        return None                          # già bilanciato: il problema è un altro
    head = t[start:last_safe] if last_safe else t[start:]
    return head.rstrip().rstrip(",") + "".join(reversed(stack))


def parse_model_json(text: str) -> tuple[Optional[dict], bool]:
    """Parsing dell'output del modello. Restituisce (oggetto, riparato).
    Ordine: parse pulito -> blocco bilanciato -> riparazione del troncamento."""
    parsed = extract_json_block(text)
    if parsed is not None:
        return parsed, False
    repaired = repair_truncated_json((text or "").strip())
    if repaired:
        try:
            return json.loads(repaired), True
        except Exception:
            return None, False
    return None, False


def parse_italian_amount(s) -> Optional[float]:
    """Converte '€ 1.500.000,00' -> 1500000.0. Restituisce None se non interpretabile."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s)
    neg = ("-" in txt) or ("(" in txt and ")" in txt)
    txt = re.sub(r"[^0-9,.\-]", "", txt)
    if not txt:
        return None
    txt = txt.replace(".", "").replace(",", ".")   # italiano: '.' migliaia, ',' decimali
    try:
        val = float(txt)
    except ValueError:
        return None
    return -val if (neg and val > 0) else val


def normalize_cup(cup) -> Optional[str]:
    """Normalizza il CUP: niente spazi, maiuscolo. Restituisce None se vuoto."""
    if not cup:
        return None
    c = re.sub(r"\s+", "", str(cup)).upper()
    return c or None


def validate_records(parsed: Optional[dict]) -> tuple[dict, list[dict]]:
    """Rende uniforme l'output del modello. Restituisce (metadati_decreto, lista_record).
    Non solleva mai eccezioni: le righe malformate vengono tenute ma segnalate."""
    parsed = parsed or {}
    meta_decreto = {
        "numero_decreto": parsed.get("numero_decreto"),
        "data_decreto": parsed.get("data_decreto"),
        "ministero_1": parsed.get("ministero_1"),
        "ministero_2": parsed.get("ministero_2"),
        "note_estrazione": parsed.get("note_estrazione"),
    }
    clean: list[dict] = []
    for r in (parsed.get("record") or []):
        if not isinstance(r, dict):
            continue
        cup = normalize_cup(r.get("cup"))
        importo = r.get("importo_eur")             # preferisco il numero del modello...
        if importo is None:
            importo = parse_italian_amount(r.get("importo_originale"))
        else:
            try:
                importo = float(importo)
            except (TypeError, ValueError):
                importo = parse_italian_amount(r.get("importo_originale"))  # ...altrimenti dall'originale
        clean.append({
            "cup": cup,
            "cup_valido": bool(cup and CUP_RE.match(cup)),
            "capitolo": (str(r["capitolo"]).strip() if r.get("capitolo") is not None else None),
            "piano_gestionale": (str(r["piano_gestionale"]).strip() if r.get("piano_gestionale") is not None else None),
            "importo_eur": importo,
            "importo_originale": r.get("importo_originale"),
            "tipo_azione": r.get("tipo_azione"),
            "riferimento_fonte": r.get("riferimento_fonte"),
            "note": r.get("note"),
        })
    return meta_decreto, clean


# =============================================================================
# SELEZIONE MODELLI E CELLE DELL'ESPERIMENTO
# =============================================================================

def select_models(models: Optional[str] = None, families: Optional[str] = None,
                  tiers: Optional[str] = None, kind: Optional[str] = None) -> list[ModelSpec]:
    """Filtra la griglia MODELS in base ai criteri CLI (liste separate da virgola).
    Il match sui modelli accetta sia lo slug sia il nome-cartella sanificato."""
    def as_set(csv: Optional[str]) -> Optional[set]:
        return {x.strip() for x in csv.split(",") if x.strip()} if csv else None

    want_models, want_fam = as_set(models), as_set(families)
    want_tier, want_kind = as_set(tiers), as_set(kind)

    selected: list[ModelSpec] = []
    for m in MODELS:
        if want_models and not (m.slug in want_models or m.dir_name in want_models):
            continue
        if want_fam and m.family not in want_fam:
            continue
        if want_tier and m.tier not in want_tier:
            continue
        if want_kind and m.kind not in want_kind:
            continue
        selected.append(m)
    return selected


def thinking_modes_for(spec: ModelSpec, requested: str) -> list[bool]:
    """Modalità di thinking VALIDE per un modello, dato ciò che si è richiesto
    ('off' | 'on' | 'both'). Restituisce una lista di booleani (True = ON).
      - OFF valido se reasoning in {none, optional}
      - ON  valido se reasoning in {optional, mandatory}
    Così un modello non-thinking non entra mai nello sweep ON (niente celle falsate)."""
    want = {"off", "on"} if requested == "both" else {requested}
    modes: list[bool] = []
    if "off" in want and spec.reasoning in ("none", "optional"):
        modes.append(False)
    if "on" in want and spec.reasoning in ("optional", "mandatory"):
        modes.append(True)
    return modes


# =============================================================================
# GESTIONE DEL RUN E I/O DEI RISULTATI (struttura a cascata)
# =============================================================================

def run_dir_path(fingerprint: str, resume_run: Optional[str]) -> Path:
    """Percorso della cartella-run: output/<timestamp>__<hash_prompt>/ (timestamp E hash,
    task 7). Con resume_run si riusa una cartella esistente. NON crea la cartella: la
    creazione avviene solo quando si scrive davvero (così il --dry-run non sporca output)."""
    if resume_run:
        return OUTPUT_DIR / resume_run
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return OUTPUT_DIR / f"{stamp}__{fingerprint}"


def result_json_path(run_dir: Path, spec: ModelSpec, pdf_stem: str, thinking_on: bool) -> Path:
    """Percorso del file di risultato per una cella (documento x modello x thinking):
    <run>/<modello_sanificato>/<documento>__think-{on|off}.json"""
    suffix = "think-on" if thinking_on else "think-off"
    return run_dir / spec.dir_name / f"{pdf_stem}__{suffix}.json"


def cell_completed_ok(path: Path) -> bool:
    """True se la cella salvata risulta completata con successo (status 'ok'). Serve al
    --resume per RIPROVARE le celle in errore invece di saltarle come se fossero fatte."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"
    except Exception:
        return False


def save_json(path: Path, payload: dict) -> None:
    """Scrittura atomica (via file .tmp) per non corrompere il risultato se il processo
    viene interrotto a metà. Crea al volo le cartelle mancanti (unico punto in cui la
    struttura a cascata prende forma su disco)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_run_manifest(run_dir: Path, specs: list[ModelSpec], requested_thinking: str) -> None:
    """Scrive run_manifest.json: griglia modelli, prezzi, parametri e condizioni del run,
    per la piena riproducibilità (cfr. paper: registrazione versione/parametri)."""
    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "prompt_fingerprint": prompt_fingerprint(),
        "price_list_date": PRICE_LIST_DATE,
        "generation": {
            "temperature": TEMPERATURE,
            # con disable_max_tokens=True il campo max_output_tokens è inerte: nel body non
            # viene inviato alcun tetto. Lo teniamo comunque a verbale per leggere i run vecchi.
            "max_output_tokens": (None if DISABLE_MAX_TOKENS else MAX_OUTPUT_TOKENS),
            "disable_max_tokens": DISABLE_MAX_TOKENS,
            "reasoning_effort_on": REASONING_EFFORT_ON,
            "requested_thinking": requested_thinking,
            "json_response_format": USE_JSON_RESPONSE_FORMAT,
        },
        "logging": {
            "log_raw_responses": LOG_RAW_RESPONSES,
            "log_dir": LOG_DIR_NAME if LOG_RAW_RESPONSES else None,
            "log_include_request": LOG_INCLUDE_REQUEST,
        },
        "models": [asdict(m) for m in specs],
    }
    save_json(run_dir / "run_manifest.json", manifest)


# =============================================================================
# LOOP PRINCIPALE DELL'ESPERIMENTO
# =============================================================================

def process(specs: list[ModelSpec], requested_thinking: str,
            limit: Optional[int], resume_run: Optional[str], dry_run: bool,
            doc_arg: Optional[str] = None) -> Path:
    """Esegue lo sweep (documento x modello x thinking) e restituisce la cartella-run.
    Con resume_run riprende una run esistente (salta le celle già completate)."""
    fingerprint = prompt_fingerprint()
    run_dir = run_dir_path(fingerprint, resume_run)
    # FLAG 2: la cartella dei log vive DENTRO la cartella-run (output/<run>/log/), così i grezzi
    # restano legati al prompt e ai parametri di quel run. Non viene creata se il flag è spento.
    log_dir: Optional[Path] = (run_dir / LOG_DIR_NAME) if LOG_RAW_RESPONSES else None
    if not dry_run:
        dump_prompt_file(run_dir, fingerprint)         # crea la cartella-run e ci scrive il prompt
        write_run_manifest(run_dir, specs, requested_thinking)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Logging integrale ATTIVO -> {log_dir}")
        if DISABLE_MAX_TOKENS:
            print("max_tokens DISATTIVATO: nessun tetto ai token di output (occhio alla spesa).")

    pdfs = select_documents(doc_arg, limit)
    if not pdfs:
        if not doc_arg:
            print(f"Nessun PDF trovato in {INPUT_DIR}")
        return run_dir

    # Avviso se qualche modello selezionato non produce celle con la modalità richiesta
    # (es. un modello non-thinking con --thinking on, o mandatory con --thinking off).
    for m in specs:
        if not thinking_modes_for(m, requested_thinking):
            print(f"NOTA: '{m.slug}' (reasoning={m.reasoning}) non ha celle valide "
                  f"per --thinking {requested_thinking}: sarà saltato.")

    n_cells = sum(len(thinking_modes_for(m, requested_thinking)) for m in specs)
    print(f"{len(pdfs)} PDF x {len(specs)} modelli -> {n_cells} celle-modello per documento\n"
          f"Run: {run_dir.name}\n")

    for pdf_path in pdfs:
        rel = pdf_path.relative_to(INPUT_DIR).as_posix()
        stem = pdf_path.stem
        try:
            text, n_pages, scanned = extract_pdf_text(pdf_path)   # ogni PDF è letto una sola volta
        except Exception as e:
            print(f"[{rel}] ESTRAZIONE TESTO FALLITA: {e}")
            continue

        if scanned or not text:
            # PDF-immagine (scansione): per ORA lo rileviamo e lo SALTIAMO.
            # Qui in futuro va innestata la gestione dei PDF-immagine, DUE OPZIONI ALTERNATIVE:
            #   OPZIONE A (vision nativa): inviare il PDF/pagina come immagine base64 ai soli modelli vision-capable (spec.vision == True).
            #   OPZIONE B (pipeline OCR): passare il PDF a Tesseract/Docling, ottenere il testo e trattarlo come un PDF nativo.
            print(f"[{rel}] sembra una scansione / vuoto ({n_pages} pag.) -> salto gli LLM.")
            continue

        messages = build_messages(text)

        for spec in specs:
            for thinking_on in thinking_modes_for(spec, requested_thinking):
                out_path = result_json_path(run_dir, spec, stem, thinking_on)
                if out_path.exists() and cell_completed_ok(out_path):
                    continue                      # resume: salto solo le celle già RIUSCITE (gli errori si riprovano)
                label = f"{spec.dir_name} [{'ON' if thinking_on else 'OFF'}]"
                if dry_run:
                    print(f"[{rel}] (dry-run) -> {label}")
                    continue

                print(f"[{rel}] -> {label}")
                entry = _run_one_cell(spec, messages, thinking_on, rel, stem, n_pages,
                                      log_dir=log_dir)
                save_json(out_path, entry)         # scrittura incrementale: non perdo lavoro pagato

    print(f"\nFatto. Risultati in: {run_dir}")
    return run_dir


def _run_one_cell(spec: ModelSpec, messages: list[dict], thinking_on: bool,
                  rel: str, stem: str, n_pages: int,
                  log_dir: Optional[Path] = None) -> dict:
    """Esegue UNA cella (documento x modello x thinking) e ne costruisce il record di
    output completo di token, costo (API e ricostruito), latenza e record estratti."""
    entry: dict = {
        "decree_file": Path(rel).name,
        "decree_relpath": rel,
        "model_slug": spec.slug,
        "model_family": spec.family,
        "model_kind": spec.kind,
        "model_tier": spec.tier,
        "thinking": "on" if thinking_on else "off",
        "reasoning_mode": spec.reasoning,
        "n_pages": n_pages,
    }
    try:
        out = call_openrouter(spec, messages, thinking_on, log_dir=log_dir, doc_stem=stem)
        parsed, repaired = parse_model_json(out["content"])
        decree_meta, records = validate_records(parsed)
        cost_recon = reconstruct_cost(spec, out["prompt_tokens"], out["completion_tokens"])
        if parsed is None:
            status = "parse_error"
        elif repaired:
            status = "ok_repaired"      # JSON troncato e richiuso: dato usabile, ma non pulito
        else:
            status = "ok"
        entry.update({
            "status": status,
            "json_repaired": repaired,
            "sent_at": out["sent_at"],
            "received_at": out["received_at"],
            "latency_seconds": out["latency_seconds"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "reasoning_tokens": out["reasoning_tokens"],
            "cached_tokens": out["cached_tokens"],
            "total_tokens": out["total_tokens"],
            "cost_usd_api": out["cost_usd_api"],
            "cost_usd_reconstructed": cost_recon,
            "finish_reason": out["finish_reason"],
            "truncated": out["finish_reason"] == "length",
            "stripped_params": out["stripped_params"],
            "response_meta": out["response_meta"],
            "decree_meta": decree_meta,
            "records": records,
            # Il grezzo si tiene se il parse FALLISCE **oppure** se riesce ma non produce
            # alcun record: "zero record" è il caso che va sempre potuto ispezionare a
            # posteriori (chiave sbagliata? array davvero vuoto? testo non estratto?).
            # Senza questo, una cella 'ok' con 0 record è indiagnosticabile e va ripagata.
            "raw_content": out["content"] if (parsed is None or not records) else None,
        })
        cost = out["cost_usd_api"]
        cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else "n/d"
        # L'etichetta segue lo STATUS reale: stampare 'ok' su una cella con parse_error
        # manda fuori strada la diagnosi (ed è successo).
        label = "ok" if entry["status"] == "ok" else entry["status"].upper()
        print(f"    {label}: {len(records)} record, "
              f"{out['prompt_tokens']}->{out['completion_tokens']} tok, "
              f"{cost_str}, {out['latency_seconds']}s")
        if entry["truncated"]:
            print("    ATTENZIONE: output troncato (alzare MAX_OUTPUT_TOKENS).")
    except Exception as e:
        entry.update({"status": "error", "error": str(e)})
        print(f"    ERRORE: {e}")
    return entry


# =============================================================================
# EXPORT & CONFRONTO (JSONL: solo stdlib; CSV: richiede pandas)
# =============================================================================

def collect_run_entries(run_dir: Path) -> list[dict]:
    """Ricarica tutte le celle salvate in una run (esclusi manifest e file .tmp)."""
    entries: list[dict] = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name == "run_manifest.json":
            continue
        try:
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"    (skip {path.name}: {e})")
    return entries


def _run_fingerprint(run_dir: Path) -> Optional[str]:
    """Fingerprint del prompt DI QUEL run (non del codice attuale): lo legge da
    run_manifest.json e, in mancanza, dal suffisso del nome cartella <timestamp>__<hash>.
    Così l'etichetta resta giusta anche con --export-only su run vecchi."""
    mani = run_dir / "run_manifest.json"
    if mani.exists():
        try:
            fp = json.loads(mani.read_text(encoding="utf-8")).get("prompt_fingerprint")
            if fp:
                return fp
        except Exception:
            pass
    return run_dir.name.split("__")[-1] if "__" in run_dir.name else None


def export_jsonl(run_dir: Path) -> None:
    """Scrive results.jsonl: UNA riga JSON per cella (l'entry completo, annidato) —
    aggregato di massa senza perdite, leggibile da pandas/DuckDB/jq. A ogni riga
    antepone `run` (nome cartella) e `prompt_fingerprint`, così unendo più run
    (glob output/*/results.jsonl) le righe restano distinguibili. Solo stdlib: è la
    fonte solida e NON dipende da pandas. Scrittura atomica via .tmp."""
    entries = collect_run_entries(run_dir)
    if not entries:
        print("Nessuna cella da esportare in JSONL.")
        return
    fingerprint = _run_fingerprint(run_dir)
    out_path = run_dir / "results.jsonl"
    tmp = run_dir / "results.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for e in entries:
            row = {"run": run_dir.name, "prompt_fingerprint": fingerprint, **e}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    print(f"JSONL scritto: {out_path.name} ({len(entries)} righe, tutte le celle)")


def export_tables(run_dir: Path) -> None:
    """Rigenera gli aggregati del run. Prima results.jsonl (fonte di massa, senza
    pandas); poi le VISTE comode CSV: runs.csv (una riga per cella), records.csv (una
    riga per record), comparison.csv (pivot importo per (decreto, CUP) x modello×thinking)."""
    export_jsonl(run_dir)   # la fonte di massa esce sempre, anche senza pandas
    try:
        import pandas as pd
    except ImportError:
        print("pandas non installato -> salto l'export CSV. `pip install pandas`.")
        return

    entries = collect_run_entries(run_dir)
    if not entries:
        print("Nessuna cella da esportare.")
        return

    run_rows, rec_rows = [], []
    for e in entries:
        run_rows.append({
            "decree_file": e.get("decree_file"),
            "model_slug": e.get("model_slug"),
            "family": e.get("model_family"),
            "kind": e.get("model_kind"),
            "tier": e.get("model_tier"),
            "thinking": e.get("thinking"),
            "status": e.get("status"),
            "n_pages": e.get("n_pages"),
            "prompt_tokens": e.get("prompt_tokens"),
            "completion_tokens": e.get("completion_tokens"),
            "reasoning_tokens": e.get("reasoning_tokens"),
            "total_tokens": e.get("total_tokens"),
            "cost_usd_api": e.get("cost_usd_api"),
            "cost_usd_reconstructed": e.get("cost_usd_reconstructed"),
            "latency_seconds": e.get("latency_seconds"),
            "n_records": len(e.get("records") or []),
            "truncated": e.get("truncated"),
            "sent_at": e.get("sent_at"),
            "error": e.get("error"),
        })
        meta = e.get("decree_meta") or {}
        for r in (e.get("records") or []):
            rec_rows.append({
                "decree_file": e.get("decree_file"),
                "model_slug": e.get("model_slug"),
                "thinking": e.get("thinking"),
                "numero_decreto": meta.get("numero_decreto"),
                "data_decreto": meta.get("data_decreto"),
                "ministero_1": meta.get("ministero_1"),
                "ministero_2": meta.get("ministero_2"),
                "cup": r.get("cup"),
                "cup_valido": r.get("cup_valido"),
                "capitolo": r.get("capitolo"),
                "piano_gestionale": r.get("piano_gestionale"),
                "importo_eur": r.get("importo_eur"),
                "importo_originale": r.get("importo_originale"),
                "tipo_azione": r.get("tipo_azione"),
                "riferimento_fonte": r.get("riferimento_fonte"),
            })

    runs = pd.DataFrame(run_rows)
    recs = pd.DataFrame(rec_rows)
    runs.to_csv(run_dir / "runs.csv", index=False, encoding="utf-8-sig")
    recs.to_csv(run_dir / "records.csv", index=False, encoding="utf-8-sig")

    # comparison.csv: importo totale per (decreto, CUP) di ogni cella, affiancati.
    # Aggrego per CUP perché i modelli differiscono nella granularità (per-PG vs aggregato).
    if not recs.empty and recs["cup"].notna().any():
        with_cup = recs[recs["cup"].notna()].copy()
        with_cup["cell"] = with_cup["model_slug"] + "::" + with_cup["thinking"]
        pivot = (with_cup.groupby(["decree_file", "cup", "cell"])["importo_eur"]
                 .sum().unstack("cell"))
        pivot["n_cells_found"] = pivot.notna().sum(axis=1)
        # Flag: tutte le celle presenti concordano sull'importo (arrotondato al centesimo)?
        value_cols = [c for c in pivot.columns if c != "n_cells_found"]
        pivot["all_agree"] = pivot[value_cols].apply(
            lambda row: len({round(v, 2) for v in row.dropna()}) <= 1, axis=1)
        pivot.to_csv(run_dir / "comparison.csv", encoding="utf-8-sig")

    # Riepilogo spesa: costo per cella-modello (somma sui documenti).
    if runs["cost_usd_api"].notna().any():
        by_cell = (runs.assign(cell=runs["model_slug"] + "::" + runs["thinking"])
                   .groupby("cell")["cost_usd_api"].sum().sort_values(ascending=False))
        print(f"\nSpesa totale (API): ${runs['cost_usd_api'].sum():.4f}")
        print("Costo per cella-modello:")
        print(by_cell.to_string())

    print(f"\nCSV scritti in {run_dir}")


# =============================================================================
# STIMA COSTO (probe: una sola cella documento × modello)
# =============================================================================

def resolve_doc(doc_arg: Optional[str]) -> Optional[Path]:
    """Sceglie IL documento per la stima (--estimate fa UNA sola chiamata).
    Senza --doc: il PRIMO PDF trovato in input/. Con --doc accetta, in quest'ordine:
    un path esistente, un percorso dentro input/, altrimenti il pattern di match_pdfs
    (token AND, o regex col prefisso 're:'). Se il pattern è ambiguo NON indovina in
    silenzio: elenca i candidati e prende il primo, dicendolo."""
    if not doc_arg:
        pdfs = find_input_pdfs(limit=1)
        return pdfs[0] if pdfs else None
    p = Path(doc_arg)
    if p.is_file():
        return p
    cand = INPUT_DIR / doc_arg
    if cand.is_file():
        return cand
    matches = match_pdfs(doc_arg)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"'{doc_arg}' corrisponde a {len(matches)} documenti; --estimate ne usa UNO solo:")
        for m in matches[:10]:
            print(f"    {m.relative_to(INPUT_DIR).as_posix()}")
        print(f"  -> uso il primo: {matches[0].name}  (restringi il pattern per cambiarlo)")
    return matches[0]


def _print_estimate(spec: ModelSpec, entry: dict, thinking_on: bool) -> None:
    """Blocco leggibile a console: costo reale, latenza, token e proiezione full-run."""
    def money(x) -> str:
        return f"${x:.6f}" if isinstance(x, (int, float)) else "n/d"
    n_docs = len(find_input_pdfs())
    cost = entry.get("cost_usd_api")
    line = "=" * 64
    print("\n" + line)
    print(f"STIMA COSTO — {spec.slug}  [thinking {'on' if thinking_on else 'off'}]  (tier {spec.tier})")
    print(line)
    print(f"  Documento:        {entry.get('decree_file')}  ({entry.get('n_pages')} pag.)")
    print(f"  Stato:            {entry.get('status')}")
    print(f"  Latenza:          {entry.get('latency_seconds')} s   (invio → risposta)")
    print(f"  Token:            {entry.get('prompt_tokens')} in → {entry.get('completion_tokens')} out"
          f"   (reasoning: {entry.get('reasoning_tokens')})")
    print(f"  Costo chiamata:   {money(cost)}   (usage.cost API OpenRouter, reale)")
    print(f"    ricostruito:    {money(entry.get('cost_usd_reconstructed'))}   (da listino {PRICE_LIST_DATE}, controprova)")
    print(f"  Record estratti:  {len(entry.get('records') or [])}")
    if isinstance(cost, (int, float)) and n_docs:
        print("-" * 64)
        print(f"  Proiezione full-run — SOLO questo modello ({spec.slug}):")
        print(f"    ~{money(cost * n_docs)}   su {n_docs} documenti in input/  (= costo chiamata × n. doc)")
        print(f"    indicativa (testo simile). NON include gli altri modelli: per il run")
        print(f"    completo sommare la proiezione di ciascun modello selezionato.")
    print(line)


def estimate_cost(slug: str, doc_arg: Optional[str], thinking_choice: str) -> None:
    """Probe di costo/latenza: UNA sola chiamata (un documento × un modello). Stampa il
    costo reale (usage.cost dell'API), la latenza già misurata dallo script e una proiezione
    della spesa su tutti i PDF in input/. Serve a stimare il costo PRIMA di un run pieno."""
    spec = next((m for m in MODELS if slug in (m.slug, m.dir_name)), None)
    if spec is None:
        print(f"Modello '{slug}' non è nella griglia MODELS (vedi --list-models per gli slug).")
        return
    # thinking risolto a UN booleano sempre valido per questo modello (niente celle falsate)
    thinking_on = (spec.reasoning == "mandatory") or (thinking_choice == "on" and spec.reasoning != "none")
    doc = resolve_doc(doc_arg)
    if doc is None:
        where = f"'{doc_arg}'" if doc_arg else f"la cartella {INPUT_DIR}"
        print(f"Nessun PDF trovato per {where}. Passa --doc <file> oppure popola input/.")
        return
    print(f"Probe: {spec.slug} [thinking {'on' if thinking_on else 'off'}] su '{doc.name}'\n")
    try:
        text, n_pages, scanned = extract_pdf_text(doc)
    except Exception as e:
        print(f"Estrazione testo fallita: {e}")
        return
    if scanned or not text:
        print(f"'{doc.name}' sembra una scansione/vuoto ({n_pages} pag.): impossibile stimare via testo.")
        return
    # --estimate non crea una cartella-run: se il logging è attivo, il grezzo finisce in
    # output/log/ (stessa convenzione, un livello più in alto).
    log_dir = (OUTPUT_DIR / LOG_DIR_NAME) if LOG_RAW_RESPONSES else None
    entry = _run_one_cell(spec, build_messages(text), thinking_on, doc.name, doc.stem, n_pages,
                          log_dir=log_dir)
    _print_estimate(spec, entry, thinking_on)


# =============================================================================
# UTILITY: verifica slug e stampa griglia
# =============================================================================

def check_models() -> None:
    """Verifica gli slug configurati contro il catalogo live di OpenRouter (GET gratuita)
    e suggerisce corrispondenze vicine per quelli non trovati."""
    print("Scarico l'elenco modelli da OpenRouter...")
    r = requests.get(MODELS_LIST_URL, timeout=60)
    r.raise_for_status()
    live = {m["id"] for m in r.json().get("data", [])}
    print(f"{len(live)} modelli disponibili.\n")
    for m in MODELS:
        if m.slug in live:
            print(f"  OK   {m.slug}")
        else:
            key = m.slug.split("/")[-1].split("-")[0].lower()
            near = sorted(x for x in live if key in x.lower())
            print(f"  MISS {m.slug}")
            if near:
                print("       vicini: " + ", ".join(near[:6]))


def list_models() -> None:
    """Stampa la griglia dei modelli configurati (per un colpo d'occhio su fasce/famiglie)."""
    print(f"{'SLUG':40s} {'FAM':10s} {'KIND':11s} {'TIER':9s} {'VIS':4s} {'REASON':10s} {'IN':>6s} {'OUT':>6s}")
    for m in MODELS:
        pin = f"{m.price_in}" if m.price_in is not None else "?"
        pout = f"{m.price_out}" if m.price_out is not None else "?"
        print(f"{m.slug:40s} {m.family:10s} {m.kind:11s} {m.tier:9s} "
              f"{'sì' if m.vision else 'no':4s} {m.reasoning:10s} {pin:>6s} {pout:>6s}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Estrazione e confronto dati dei decreti via OpenRouter.")
    ap.add_argument("--check-models", action="store_true",
                    help="Verifica gli slug contro il catalogo live di OpenRouter ed esci.")
    ap.add_argument("--list-models", action="store_true",
                    help="Stampa la griglia dei modelli configurati ed esci.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Processa solo i primi N PDF (run di prova economica).")
    ap.add_argument("--models", type=str, default=None,
                    help="Slug (o nomi-cartella) separati da virgola: sottoinsieme di MODELS.")
    ap.add_argument("--families", type=str, default=None,
                    help="Filtra per famiglia, es. 'anthropic,openai'.")
    ap.add_argument("--tiers", type=str, default=None,
                    help="Filtra per fascia, es. 'small,frontier'.")
    ap.add_argument("--kind", type=str, default=None,
                    help="Filtra per tipo: 'commercial' o 'open'.")
    ap.add_argument("--thinking", choices=["off", "on", "both"], default=THINKING_DEFAULT,
                    help=f"Condizione thinking (default dal codice: {THINKING_DEFAULT}); off, on, oppure both.")
    ap.add_argument("--resume", type=str, default=None,
                    help="Nome della cartella-run da riprendere (salta le celle già fatte).")
    ap.add_argument("--export-only", type=str, default=None,
                    help="Non chiama le API: rigenera JSONL + CSV dalla cartella-run indicata.")
    ap.add_argument("--estimate", type=str, default=None, metavar="SLUG",
                    help="Stima costo/latenza: UNA chiamata (un documento × il modello SLUG) e "
                         "stampa costo reale, latenza e proiezione full-run. NON salva un run.")
    ap.add_argument("--doc", type=str, default=None, metavar="PATTERN",
                    help="Seleziona i documenti (vale sia per il run sia per --estimate). "
                         "Accetta un path, oppure token separati da spazio in AND sul percorso "
                         "sotto input/ (es. --doc \"270 2023\"), oppure una regex col prefisso "
                         "'re:' (es. --doc \"re:dm[ _-]?270\"). Nel run processa TUTTI i match; "
                         "con --estimate ne usa uno solo.")
    ap.add_argument("--no-max-tokens", action="store_true",
                    help="Non invia 'max_tokens': nessun tetto lato script (override di "
                         "DISABLE_MAX_TOKENS). Occhio alla spesa: sparisce l'unico freno per chiamata.")
    ap.add_argument("--max-tokens", type=int, default=None, metavar="N",
                    help=f"Alza/abbassa il tetto per questo run (override di MAX_OUTPUT_TOKENS, "
                         f"attualmente {MAX_OUTPUT_TOKENS}). Ignorato se c'è --no-max-tokens.")
    ap.add_argument("--log-raw", action="store_true",
                    help="Attiva il logging integrale delle risposte per questo run "
                         "(override di LOG_RAW_RESPONSES).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Elenca le celle che verrebbero eseguite senza chiamare le API.")
    return ap


def apply_cli_overrides(args) -> None:
    """Sovrascrive i flag globali con quanto passato da riga di comando. Gli override valgono
    per TUTTO il processo (run, --estimate) e finiscono nel run_manifest.json: un run resta
    quindi ricostruibile anche se i default nel codice cambiano in seguito.
    Precedenza: --no-max-tokens batte --max-tokens (togliere il tetto rende il valore inerte)."""
    global DISABLE_MAX_TOKENS, MAX_OUTPUT_TOKENS, LOG_RAW_RESPONSES
    if args.no_max_tokens:
        DISABLE_MAX_TOKENS = True
        if args.max_tokens is not None:
            print("NOTA: --max-tokens ignorato perché è attivo --no-max-tokens.")
    elif args.max_tokens is not None:
        if args.max_tokens <= 0:
            print(f"--max-tokens deve essere positivo (ricevuto {args.max_tokens}): ignorato.")
        else:
            MAX_OUTPUT_TOKENS = args.max_tokens
    if args.log_raw:
        LOG_RAW_RESPONSES = True


def main() -> None:
    args = build_arg_parser().parse_args()
    apply_cli_overrides(args)   # prima di qualsiasi dispatch: vale anche per --estimate

    if args.check_models:
        check_models()
        return
    if args.list_models:
        list_models()
        return
    if args.export_only:
        export_tables(OUTPUT_DIR / args.export_only)
        return
    if args.estimate:
        estimate_cost(args.estimate, args.doc, args.thinking)
        return

    specs = select_models(args.models, args.families, args.tiers, args.kind)
    if not specs:
        print("Nessun modello selezionato con i filtri indicati (vedi --list-models).")
        return

    run_dir = process(specs, args.thinking, args.limit, args.resume, args.dry_run, args.doc)
    if not args.dry_run:
        export_tables(run_dir)


if __name__ == "__main__":
    main()