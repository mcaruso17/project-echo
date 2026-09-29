# Estrazione dati decreti PA con LLM (OpenRouter)

> **Ripartire dal lavoro in corso: [`STATO_LAVORI.md`](STATO_LAVORI.md)** — obiettivo,
> risultati misurati, decisioni prese, bug noti e i prossimi passi in ordine.

Confronto tra più LLM sull'estrazione di CUP / capitolo / piano gestionale / importo
da decreti in PDF. Stesso prompt e stesso testo per tutti i modelli.

- **`mcex.py`** — l'esperimento (versione corrente; `newex.py` ed `estrattore.py` sono i predecessori).
- **`confronto.py`** — precision/recall/F1 contro il benchmark manuale `Output a MANO.xlsx`.
- **`punteggio.py`** — **exact set match per documento**: il criterio d'arresto del ciclo.
- **`ciclo.py`** — l'orchestratore del ciclo di miglioramento del prompt.
- **`giudice.py`** — il revisore terzo (TPLLM) che analizza le divergenze e propone correzioni.
- **`sonda.py`** — una chiamata "ciao" che salva il JSON grezzo di OpenRouter (per capirne lo schema).
- **`input/`** — i PDF (letti in modo ricorsivo). 
- **`output/`** — una cartella per ogni run.
- **`documenti_eleggibili.txt`** — i documenti coperti dal benchmark, da passare a `--docs-file`.

## Chiave API
La chiave si legge **solo** dall'ambiente, senza ripieghi nel sorgente (questa cartella è
sincronizzata e condivisa: una chiave scritta qui è una chiave pubblicata):
```
export OPENROUTER_API_KEY="sk-or-v1-..."
```

## Prompt iniettabile (per il ciclo di miglioramento)
Il prompt non vive più solo dentro `mcex.py`: lo si può passare da file, e il fingerprint
si ricalcola da sé, quindi ogni variante finisce in una cartella-run distinta.
```
python mcex.py --print-prompt > cicli/prompts/v000.json     # semina: round-trip esatto
python mcex.py --prompt-file cicli/prompts/v001.json --docs-file documenti_eleggibili.txt
python punteggio.py <cartella_run> --dettaglio               # quanti documenti esatti, e cosa manca
```
Il file è `{"version", "parent", "system", "user_template"}`. `user_template` **deve**
contenere `<<DECREE_TEXT>>` esattamente una volta: se manca, lo script si ferma prima di
spendere invece di inviare 16 chiamate prive del testo dell'atto.

`--resume` rifiuta di riprendere una run creata con un prompt diverso (`--force-prompt-mismatch`
per forzare): altrimenti due prompt finirebbero nella stessa cartella e nello stesso manifest.

## Output
Ogni run è una cartella `output/<timestamp>__<hash_prompt>/` autocontenuta.

- **Fonte per-cella**: `<modello>/<documento>__think-{on,off}.json`, un file per cella
  `(documento × modello × thinking)`, scritto in modo atomico (regge il `--resume`).
- **`results.jsonl`** — l'aggregato di massa: **una riga JSON per cella**, l'entry completo
  e annidato (record, token, costo, latenza, `status`). Non perde nulla. Ogni riga porta
  anche `run` e `prompt_fingerprint`, così più run si uniscono senza ambiguità. È la fonte
  per l'analisi; esce sempre, anche senza pandas.
- **`runs.csv` / `records.csv` / `comparison.csv`** — viste comode per Excel, derivate e
  lossy (richiedono pandas). Non sono la fonte.

Analisi di massa (SQL diretto, niente DB):
```
duckdb -c "SELECT model_slug, thinking, sum(cost_usd_api) c FROM read_json_auto('output/*/results.jsonl') GROUP BY 1,2 ORDER BY c DESC"
```

## Dipendenze
```
pip install requests pdfplumber pandas
```

## Costo
Costo **autorevole** = `usage.cost` di OpenRouter (campo `cost_usd_api`), sempre aggiornato.
I prezzi di listino nella griglia servono solo alla ricostruzione da token
(`cost_usd_reconstructed`) e possono invecchiare.

Per una stima **prima** di un run pieno: `--estimate <slug>` fa una sola chiamata (un
documento × un modello) e stampa costo reale, latenza (invio→risposta) e la proiezione
sul numero di PDF in `input/`. La proiezione è **solo per quel modello** (non la somma di
tutti): per il run completo, sommare la proiezione di ciascun modello. Non salva alcun run.
]
## Prima di un run vero: congelare gli slug
Gli ID dei modelli su OpenRouter cambiano. Verificarli (chiamata gratuita) e correggere
in `estrattore.py` (lista `MODELS`) le righe segnate `MISS`:
```
python estrattore.py --check-models
```

## Comandi
```
python estrattore.py --list-models                              # griglia modelli
python estrattore.py --estimate <slug>                          # stima costo/latenza: 1 chiamata (1° PDF x modello)
python estrattore.py --estimate <slug> --doc <file>             # stima su un documento specifico
python estrattore.py --dry-run --thinking both --limit 1        # anteprima celle, zero costi
python estrattore.py --limit 1 --families anthropic             # run vero economico
python estrattore.py --thinking both                            # esperimento completo (ON e OFF)
python estrattore.py --resume <cartella_run> --dry-run          # cosa resta da fare, zero costi
python estrattore.py --resume <cartella_run> --thinking both    # riprende: rifà solo error + mancanti
python estrattore.py --export-only <cartella_run>               # rigenera JSONL + CSV
python sonda.py                                                 # JSON grezzo di una chiamata base
python mcex.py --print-prompt                                   # prompt attuale in JSON, per --prompt-file
python mcex.py --prompt-file p.json --docs-file d.txt           # run con prompt e documenti scelti
python mcex.py --print-run-dir ...                              # path della run come ultima riga (orchestratore)
python punteggio.py <run> --dettaglio                           # exact match per documento + FN/FP
```
Filtri combinabili: `--models --families --tiers --kind --limit --thinking {off,on,both}`

## Riprendere un run interrotto (credito esaurito, Ctrl+C, crash)
Niente va perso: ogni cella è scritta su disco **appena riesce**, in modo atomico. Un'interruzione
non corrompe i risultati già salvati.

- Le celle riuscite hanno `status: "ok"`; le fallite `status: "error"` (messaggio in `error`).
  Quando il credito finisce OpenRouter risponde **402/429** → quelle celle vengono salvate come
  `error`, **non** `ok`.
- **`--resume` salta SOLO le celle `ok`** e riprova le `error` + le mancanti: non ripaghi mai il
  lavoro già riuscito.

Ripresa senza sprechi:
1. Ricarica il credito su OpenRouter.
2. Guarda **quanto manca senza spendere**: `--resume <cartella> --dry-run` elenca solo le celle da fare.
3. Riprendi: `--resume <cartella>` con **gli stessi filtri** dell'originale.

Regole per non buttare soldi:
- **Stessi filtri del run originale** (`--thinking`, `--models/--families/--tiers`, `--limit`): il
  `--resume` ricalcola le celle dai filtri CLI *di questa* invocazione, non dal manifest. Senza
  filtri processa TUTTI i modelli → aggiungeresti celle nuove (e costo).
- **Non cambiare il prompt** tra run e ripresa: cambierebbe l'`hash_prompt` e le celle dentro la
  stessa cartella (che porta l'hash originale nel nome) diventerebbero incoerenti.
- Se il processo è morto prima di generare gli aggregati, rigenerali con `--export-only <cartella>`
  e ispeziona `status`/`error` in `runs.csv` o `results.jsonl`.


---

# Ciclo di miglioramento del prompt

Un round: **estrazione → punteggio → controllo regressioni → revisore → ATTESA APPROVAZIONE
→ applicazione**. Il ciclo si ferma **sempre** al gate: nessuna modifica al prompt entra in
produzione senza che una persona l'abbia letta.

```
python ciclo.py --loop L1 init          # semina v000, scrive config.json e state.json
python ciclo.py --loop L1 run           # avanza fino al gate e si ferma
python ciclo.py --loop L1 stato         # storia dei round, costo, migliore
python ciclo.py --loop L1 approva 1 --si
python ciclo.py --loop L1 approva 1 --edit                  # $EDITOR sul candidato, poi promuove
python ciclo.py --loop L1 approva 1 --no --motivo "..."     # riparte dal MIGLIORE, non dal bocciato
```

## Criterio d'arresto
Un documento è **risolto** quando l'insieme delle coppie (CUP, importo) estratte coincide
esattamente con quello di Alessandro — zero mancanti, zero in più, tolleranza ±1 EUR.
Dove il benchmark ha compilato anche **CAPITOLO/PG** e **TIPOLOGIA** (assegnato, impegnato,
rimodulato, revocato, ...), la chiave diventa **(CUP, importo, capitolo/PG, tipologia)**: è
il livello pieno, il first best — collegare fonte dell'impegno, sostanza e progetto.
`punteggio.py` stampa accanto a ogni cella quanti documenti ha risolto a quel livello.
Proiezione: solo i record `livello_importo == "totale"`, importo diviso per il numero di
CUP (è il livello `pairtot` che `confronto.py` già calcola; le altre tre proiezioni sono
state misurate e rendono molto peggio).

Lo stop scatta solo per un modello **open**: la cella commerciale è un soffitto di
riferimento e non può farlo scattare, altrimenti il ciclo si fermerebbe proprio prima di
rispondere alla domanda che lo motiva.

## Il revisore, e cosa NON può fare
`giudice.py` classifica ogni divergenza in una di quattro cause — `modifica_prompt`,
`rilievo_estrazione`, `rilievo_benchmark`, `rilievo_capacita` — e **solo la prima**
autorizza a toccare il prompt. Le altre diventano segnalazioni per una persona.

I vincoli sono **verificati automaticamente**, non raccomandati: sezione 4 (FORMATO DI
OUTPUT) e regola R6 intoccabili, massimo 3 modifiche per round, nessun CUP / anno /
importo / numero di decreto dentro il testo di una regola (anti-overfitting), lunghezza
del system prompt fra 0,70× e 1,10× (accorciare sì, allungare quasi mai: i modelli piccoli
peggiorano con i prompt lunghi), e coerenza fra la lista di modifiche dichiarata e il
prompt materializzato. Una violazione scarta la proposta **prima** che arrivi
all'approvazione.

Gli esempi della sezione 5 sono l'unica eccezione all'anti-overfitting — sono casistici per
natura — ma possono essere **aggiunti, mai modificati**.

## Protezione dalle regressioni
`frozen_solved` tiene i documenti risolti da qualunque cella in qualunque round approvato.
Se un round ne fa regredire uno, l'approvazione richiede `--accetto-regressione`
esplicito. La tabella vinti/persi per cella è la prima cosa in `proposta.md`, sopra la
prosa del revisore.

## Stato su disco
```
cicli/<loop>/
  config.json  state.json  scala_hardware.csv
  prompts/v000.json v001.json ...        # immutabili
  candidati/vNNN_candidate.json
  rounds/rNNN/  run_dir.txt  mcex.log  scores.json
                contesto_giudice.json  proposta.json  proposta.md
```
`scala_hardware.csv` accumula una riga per (round × modello): è il deliverable che risponde
alla domanda istituzionale — *quale gradino hardware basta?* — e si popola gratis.
