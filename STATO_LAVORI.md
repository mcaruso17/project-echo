# Stato dei lavori — ciclo di miglioramento del prompt

**Sessione del 24 settembre 2026.** Documento di ripresa: chi riapre il lavoro (persona o
assistente) dovrebbe poter ripartire da qui senza rileggere nulla.

---

## 1. L'obiettivo, e perché non è "quale modello vince"

Estrarre dai decreti italiani (DM / DI / DD / delibere CIPESS) le azioni di finanziamento
che legano risorse (capitolo, piano gestionale) a progetti identificati da CUP.

**La domanda di ricerca è: un modello a pesi aperti, installabile su macchine del
Ministero, può fare questo lavoro al livello di un'analista esperta?** Non "qual è il
modello migliore". Ne discendono tre conseguenze che attraversano tutto:

1. I modelli commerciali sono **soffitto di riferimento**, una sola cella. Il prompt si
   ottimizza per i modelli aperti.
2. Il criterio d'arresto scatta **solo per un modello open**.
3. L'hardware non è deciso e **il trial deve informare l'acquisto**: la dimensione del
   modello è una variabile da riportare, non un vincolo da subire.

Il metro di misura è il lavoro manuale di **Alessandro** (`Output a MANO.xlsx`): 272 righe,
3 colonne (DOCUMENTO / CUP / IMPORTO), 17 documenti sui 25 in `input/`.

---

## 2. Dove siamo: i numeri

Baseline misurato il 24/09 — run `output/20260924-122028__d02c735d`, prompt **v000
invariato** (`d02c735d`), 16 documenti, 10 celle, $9,26 spesi.

| Modello | Gradino hardware | VRAM ~4-bit | Document Arena | **solved/16** | pairtot F1 | cup F1 | $/round |
|---|---|---|---|---:|---:|---:|---:|
| `anthropic/claude-opus-5.5` | soffitto | — | famiglia #1 | **13/16** | 0,972 | 0,996 | 4,97 |
| `moonshotai/kimi-k3` | oltre il nodo | ~1,5 TB | assente | **12/16** | 0,961 | 0,996 | 2,63 |
| `deepseek/deepseek-v4-flash` | **2-4 GPU** | ~160 GB | assente | **11/16** | 0,846 | 0,922 | **0,04** |
| `deepseek/deepseek-v4-pro-0813` | oltre il nodo | ~900 GB | assente | 10/16 | 0,827 | 0,994 | 0,59 |
| `google/gemma-4-31b-it` | **1 GPU** | ~20 GB | #35 (1425) | **9/16** | 0,807 | 0,991 | 0,11 |
| `minimax/minimax-m3` | da accertare | ignota | #32 (1435) | 9/16 | 0,777 | 0,996 | 0,19 |
| `moonshotai/kimi-k2.6` | 8 GPU | ~550 GB | **#27 (1451)** | 8/16 | 0,760 | 0,994 | 0,54 |
| `google/gemma-4-26b-a4b-it` | 1 GPU | ~16 GB | assente | 5/16 | 0,594 | 0,784 | 0,05 |
| `qwen/qwen3.6-35b-a3b` | 1 GPU | ~20 GB | assente | 5/16 | 0,362 | 0,946 | 0,09 |
| `z-ai/glm-4.7-flash` | 1 GPU | ~18 GB | assente | 4/16 | 0,321 | 0,991 | 0,06 |

**Il gradino da una sola GPU è vivo**: `gemma-4-31b` (31B densi, Apache 2.0, ~20 GB) fa
9/16 con un prompt scritto per modelli di frontiera e mai adattato. Il gradino 2-4 GPU è
già a 11/16 per quattro centesimi a round.

### Tre risultati che vale la pena non dimenticare

**a) Document Arena non predice questo compito.** `kimi-k2.6` è #27, il miglior open della
classifica, e qui fa 8/16 — sotto `deepseek-v4-flash`, che nella classifica **non compare
affatto** e fa 11/16. Scegliere i candidati dall'arena avrebbe tenuto il modello sbagliato
e scartato il migliore. La classifica è voto umano su Q&A documentale generica, non
estrazione strutturata con match esatto: va usata come indizio, mai come verdetto.

**b) Il collo di bottiglia non è leggere i documenti: è la convenzione totale/quota.**
`cup_F1` è ≈ 0,99 per quasi tutti, anche per chi fa 4/16: **trovano i CUP giusti**. Quello
che sbagliano è etichettare l'importo.

| Modello | % record marcati "totale" | solved |
|---|---:|---:|
| opus-5.5 | 72% | 13/16 |
| kimi-k3 | 72% | 12/16 |
| deepseek-v4-flash | 87% | 11/16 |
| gemma-4-31b | 64% | 9/16 |
| minimax-m3 | 50% | 9/16 |
| glm-4.7-flash | 39% | 4/16 |
| qwen3.6-35b | 22% | 5/16 |

La correlazione è netta. È un problema di **istruzioni**, non di capacità: c'è margine
reale senza cambiare hardware. Il revisore è arrivato alla stessa diagnosi per conto suo.

**c) I costi sono asimmetrici.** I sette candidati installabili costano **$1,56 per round
in tutto**; i due soffitti più l'ancora storica ne costano 7,37. L'83% della spesa serve a
capire *perché* un candidato fallisce, non a farlo vincere.

---

## 3. Cosa resta aperto, e perché è quasi tutto in mano ad Alessandro

`opus-5.5` fallisce **tre** documenti, e nessuno è un errore di lettura:

| Documento | Divergenza | Natura |
|---|---|---|
| `2022 DI Mims Mef 97` | stesso € 180.000.000 su due CUP diversi: `B31F20000030005` (Alessandro) vs `B44D20000040001` (modello, che cita la riga Milano M1 Baggio-Olmi-Valsesia) | **arbitrato** — Q4 |
| `2023 DM MIT 342-2023` | Alessandro ha 9 righe per `H49J21005480003`, il modello 1 (colonna "Centro-Nord Potenziamento") | **convenzione** — Q2 |
| `2025 DD MIT 141-2025` | il modello **ha trovato** € 2.200.000.000 ma l'ha marcato `quota`; il filtro lo perde e promuove a errore i suoi tre `totale` | **convenzione** — Q3 |

Più uno **in quarantena**, fuori dal denominatore: `2024 delibera CIPE del 19 dicembre
2024` → `content_filter`, 0 record, riproducibile ad agosto e a settembre. Il PDF è uno
scrape di Gazzetta Ufficiale su due colonne: l'estrazione le interlaccia e infila nel testo
un'autorizzazione AIFA su farmaci, che fa scattare il filtro del provider. **Nessun prompt
lo risolve**: serve il ritaglio a due colonne in `extract_pdf_text()` (mcex.py:524-549),
con `page.crop()` di pdfplumber o `get_text("blocks")` di pymupdf ordinato per
`(x0 // half_width, y0)`. ~40 righe, non ancora fatto.

> **16/16 è raggiungibile, ma non è il prompt a deciderlo: sono le risposte di Alessandro.**
> Senza Q2 e Q3 nessun numero di round converge — si chiederebbe a un modello di indovinare
> una convenzione non scritta, e l'unico modo di "riuscirci" sarebbe l'overfitting.

### La scheda per Alessandro — è il prossimo passo
`convenzioni_da_chiarire.md` (le domande) + `convenzioni_da_chiarire.xlsx` (sei fogli con
le righe in conflitto affiancate e il testo del decreto).

- **Q1 CUP Master** — *solo una conferma*: tutti e dieci i modelli mettono il codice
  `(CUP MASTER)` nel campo CUP, come Alessandro. È **la nostra istruzione** a dire il
  contrario, e va corretta.
- **Q2 colonne multiple** (`342-2023`) — una riga per colonna valorizzata, o una sola?
- **Q3 totale vs quota** (`141-2025`) — **la domanda che pesa di più**, vedi §2b.
- **Q4** — il CUP da 180 milioni: arbitrato diretto.
- **Q5** — gli 8 documenti senza righe: conferma che non contengano azioni con CUP.
- Segnalazioni minori: una riga duplicata in `344-2023`; `PROV0000026838` (14 caratteri).

---

## 4. Il sistema costruito

```
mcex.py        estrattore (già esistente, esteso)
confronto.py   precision/recall/F1 (già esistente, corretto)
punteggio.py   NUOVO  exact set match per documento = criterio d'arresto
giudice.py     NUOVO  revisore terzo (TPLLM)
ciclo.py       NUOVO  orchestratore, macchina a stati, gate di approvazione
```

### `mcex.py` — cosa è stato aggiunto
- `--print-prompt` / `--prompt-file` — il prompt esce ed entra in JSON
  `{version, parent, system, user_template}`. **Round-trip esatto**: `--print-prompt |
  --prompt-file` riproduce il fingerprint `d02c735d`.
- **Due validazioni che fermano prima di spendere**: `<<DECREE_TEXT>>` presente esattamente
  una volta, e `system` di lunghezza plausibile. Senza, un prompt malformato manderebbe
  16 chiamate prive del testo dell'atto, a prezzo pieno.
- `--docs-file` — restringe il run a un elenco di stem.
- `--print-run-dir` — path della run come ultima riga di stdout (contratto con l'orchestratore).
- **Guardia su `--resume`**: `run_dir_path()` ignorava il fingerprint quando si riprendeva
  una run. Col prompt iniettabile sarebbe stato un vettore di corruzione silenziosa — due
  prompt, una cartella, un manifest. Ora si rifiuta, con `--force-prompt-mismatch` come
  via d'uscita esplicita.
- `--workers N` (default 1) — celle in parallelo dentro ogni documento. Sicuro: path di
  output distinti per costruzione, scrittura atomica, `messages` di sola lettura. Con 4
  worker il round passa da ~3 ore a ~45 minuti. Aggiunto **jitter** al backoff, altrimenti
  i worker ritentano all'unisono dopo un 429.
- Prezzi riallineati al catalogo live (erano fermi al 23/07: `gpt-5.6-sol` era passato da
  $5/$30 a $2/$10). Aggiunti 5 modelli: `claude-opus-5.5`, `minimax-m3`, `kimi-k2.5`,
  `gemma-4-31b-it`, `gemma-4-26b-a4b-it`.

### `confronto.py` — correzioni
- Bug `:147`: `decree_file[:-4]` tagliava 4 caratteri incondizionatamente mentre `:73`
  controllava l'estensione. Unificato in `doc_key()`.
- **CUP Master e provvisori recuperati**: il prompt li tiene in campi distinti, il
  benchmark li mette nella colonna CUP. Senza il recupero quei record erano invisibili al
  confronto e certi documenti insolubili per costruzione. Questa correzione da sola ha
  risolto il falso negativo da **€ 631.373.163** su `97/2022`.

### `punteggio.py` — il criterio d'arresto
```
solved(d) ⟺ insieme delle coppie (CUP, importo) identico a quello di Alessandro,
            zero mancanti, zero in più, tolleranza ±1 EUR
proiezione: solo livello_importo == "totale", importo diviso per il numero di CUP
STOP quando un modello OPEN raggiunge il totale dei documenti eleggibili,
     E il risultato si riproduce in un re-run della stessa cella
```
**La proiezione è stata scelta misurando**, non supponendo. Le quattro combinazioni di
(filtro su `livello_importo`) × (dividere o no):

| proiezione | fable-5 | deepseek-v4-pro |
|---|---|---|
| **solo "totale", diviso** | **13/17** | **9/17** |
| solo "totale", non diviso | 7/17 | 5/17 |
| tutti i record, diviso | 9/17 | 7/17 |
| tutti i record, non diviso | 4/17 | 4/17 |

La prima vince nettamente e coincide col livello `pairtot` che `confronto.py` già calcola.
Prendere tutti i record fa esplodere i falsi positivi: le quote annuali entrano in
concorrenza con i totali.

**Denominatore**: solo i documenti con benchmark **effettivamente eseguiti**, meno la
quarantena. Un documento non processato non è un fallimento, è un'assenza.

### `giudice.py` — il revisore terzo
Non è un riscrittore di prompt: è un **classificatore di cause**. Ogni divergenza finisce
in una di quattro categorie, e **solo la prima autorizza a toccare il prompt**:

| causa | conseguenza |
|---|---|
| `modifica_prompt` | istruzioni ambigue → si corregge |
| `rilievo_estrazione` | il testo non è mai arrivato al modello → nessun prompt lo risolve |
| `rilievo_benchmark` | l'evidenza contraddice Alessandro → va portato a lei, non corretto |
| `rilievo_capacita` | sbagliano i piccoli e non i grandi → è il dato per l'acquisto |

Senza questa separazione il revisore travestirebbe da regola un bug di estrazione,
peggiorando un prompt che vale F1 0,97.

**Contesto**: prompt integrale, digest dei fallimenti per cella, e **finestre di evidenza**
(±1500 caratteri attorno a ogni occorrenza del CUP, più il blocco `[TABELLA]` che lo
contiene; tetti 3k per CUP, 20k per documento, 60k totali). Un PDF da 25 pagine non entra
mai intero. Se un CUP non compare nel testo estratto si emette
`CUP ASSENTE DAL TESTO ESTRATTO`: è il segnale che serve un rilievo di estrazione.

**I vincoli sono validati, non raccomandati** — provati con dieci input avvelenati:
sezione 4 (FORMATO DI OUTPUT) e regola R6 intoccabili; massimo 3 modifiche; nessun CUP /
anno / importo / numero di decreto nel testo di una regola; lunghezza fra 0,70× e 1,10×
(accorciare sì, allungare quasi mai: i modelli piccoli peggiorano coi prompt lunghi);
coerenza fra lista di modifiche e prompt materializzato; e la **regola di autorità** — un
documento classificato come non risolvibile dal prompt non può comparire fra quelli che una
modifica promette di migliorare.

Gli esempi della sezione 5 sono l'unica eccezione ai letterali (sono casistici per natura)
ma possono essere **aggiunti, mai modificati**.

### `ciclo.py` — l'orchestratore
`SEED → EXTRACT → SCORE → GATE_REGRESSIONE → GIUDICE → ATTESA_APPROVAZIONE → APPLICA`

Stato interamente su disco, ogni fase ripartibile. Non re-implementa nulla: chiama
`mcex.py` e `confronto.py` come sottoprocessi e legge `punteggio.py`.

- **Lo stop scatta solo per un modello open.** La cella commerciale misura ma non decide.
- **Su rifiuto si riparte da `best_version`**, non dal candidato bocciato.
- **`--edit` rivalida anche quello che scrivi tu**: se la tua correzione a mano viola un
  vincolo, non viene promossa.
- **Regressioni**: `frozen_solved` tiene i documenti risolti da qualunque cella in
  qualunque round approvato; per accettarne la perdita serve `--accetto-regressione`.
- `scala_hardware.csv` accumula una riga per round × modello: è il deliverable che risponde
  a *quale gradino hardware basta*.

---

## 5. Bug trovati girando il sistema per davvero

Nessuno dei tre sarebbe emerso compilando soltanto.

1. **La regex anti-overfitting non riconosceva nessun CUP** — contava 10 cifre finali
   invece di 11. Era un validatore silenziosamente inutile: avrebbe lasciato passare un
   prompt ritagliato sui documenti del test senza che nessuno se ne accorgesse.
2. **Il denominatore contava come falliti i documenti mai eseguiti.** Nel round di prova il
   revisore ha ricevuto 14 documenti come se il modello avesse sbagliato. Se n'è accorto da
   solo e ha protestato che l'evidenza era vuota — il marcatore previsto per i PDF
   scansionati ha fatto da sentinella per un'assenza di tutt'altra natura.
3. **`approva --si` ha promosso un candidato residuo** di un tentativo precedente dello
   stesso round, benché il revisore stavolta non avesse proposto nulla. Sarebbe entrato in
   produzione un prompt che nessuno aveva approvato. Ora tre guardie: validazione superata,
   modifiche effettivamente proposte, e `parent` coerente col prompt in uso.

---

## 6. Decisioni prese, da non rimettere in discussione

| Decisione | Scelta |
|---|---|
| Autonomia del revisore | **propone, Matteo approva** ogni round |
| Ampiezza del champion set | **larga**: 7 open, tutti i gradini |
| Criterio d'arresto | **exact set match** per documento, solo per un modello open |
| Layout del codice | nuovo orchestratore che **riusa** mcex e confronto |
| Soffitto commerciale | **`claude-opus-5.5`** ($4/$20), non fable-5. Baseline ripartita da capo |
| Kimi K3 | **solo nel baseline**, riattivabile quando serve un `rilievo_capacita` |
| Chiave API | **resta nel sorgente**: è condivisa fra i tre del gruppo che lavorano in questa cartella |

Memoria fresca a ogni tentativo: già vero e da preservare — `build_messages()` (mcex.py:473)
costruisce due messaggi nuovi per documento, `temperature = 0.0`, nessuno stato fra chiamate.

---

## 7. Come si riparte

```bash
cd "…/5. LLMs"

# 1. il ciclo vero, dopo le risposte di Alessandro
python ciclo.py --loop L1 init        # champion a 9 celle (cicli/L1/champion_round.txt), ~$5,91/round
python ciclo.py --loop L1 run         # ~45 min, si ferma al gate
python ciclo.py --loop L1 stato
python ciclo.py --loop L1 approva 1 --edit      # <- vedi raccomandazione sotto

# strumenti singoli
python punteggio.py <run> --dettaglio           # exact match + FN/FP per documento
python confronto.py <run>                       # precision/recall/F1
python mcex.py --check-models                   # gratuito: verifica gli slug
```

**Raccomandazione per il round 1.** Le risposte a Q2 e Q3 conviene **scriverle a mano** nel
prompt v001 con `approva --edit`, invece di lasciarle dedurre al revisore. Sono convenzioni
che Alessandro avrà appena dichiarato: dedurle è un giro inutile, e il revisore lavora
meglio sulle divergenze che restano *dopo* che le regole note sono nero su bianco.

### Cosa resta da fare, in ordine
1. **Vedere la scheda con Alessandro** — sblocca tutto il resto.
2. Scrivere v001 a mano con le sue risposte, e girare il round 1.
3. **Ritaglio a due colonne per la Gazzetta Ufficiale** (mcex.py:524-549): toglie la
   delibera CIPE dalla quarantena e porta il denominatore da 16 a 17. Verifica: il testo
   estratto non deve più contenere il frammento AIFA.
4. Un round con `--thinking on` sulle celle open: al Ministero si paga in tempo di calcolo,
   non in denaro, quindi la convenienza è diversa da quella di un servizio a consumo.
5. Accertare i **parametri di `minimax-m3`**: è l'unica riga della scala con il gradino
   hardware ignoto, e fa 9/16.
6. `looks_scanned` (mcex.py:130) fa la media sull'intero file: in `286-2023` quattro pagine
   buone mascherano 17 vuote e il file passa mutilato. Va reso per pagina. **Bassa
   priorità**: quel documento ha una sola riga di benchmark e il modello la azzecca già.
7. OCR per `286-2023` — richiede `brew install tesseract tesseract-lang` (non installati).
   Priorità più bassa ancora.
8. Alla convergenza o al plateau: congelare il prompt, poi la griglia completa a 19 modelli
   (una volta sola, $80-150; `gpt-5.5-pro` da solo può valere $40+, decidere se includerlo).
9. **Relazione finale**: qualità contro gradino hardware, con in evidenza l'avvertenza sui
   MoE — *la memoria la fissano i parametri totali, non quelli attivi*. Kimi K2.6 attiva 32B
   per token ma i pesi devono stare tutti in memoria: è un nodo da 8 GPU, non una
   workstation. È l'equivoco più costoso possibile in una gara d'appalto.

### Note sparse
- `duckdb` **non è installato**: la one-liner di analisi documentata nel README non gira.
- `output/20260924-181513__d02c735d` è un run di **prova** (2 documenti, 1 modello) con lo
  stesso fingerprint del baseline: attenzione a non confonderli quando si fa un glob.
- Il ciclo `cicli/TEST/` è la verifica end-to-end, tenuto come documentazione.
- I due script in `0. Backup/` (`openrouter check.py`, `openrouter_check_V2_superseded.py`)
  contengono ancora la chiave in chiaro: lasciati intatti perché preesistenti.
- Il piano completo che ha guidato il lavoro è in
  `~/.claude/plans/we-need-to-work-concurrent-spring.md`.
