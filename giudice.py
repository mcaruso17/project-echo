#!/usr/bin/env python3
"""
giudice.py — il revisore terzo (TPLLM) del ciclo di miglioramento del prompt.

COSA FA, E SOPRATTUTTO COSA NON FA.
Guarda le divergenze fra quello che estraggono i modelli e il benchmark manuale di
Alessandro, va a rileggere il testo del decreto nei punti contestati, e classifica OGNI
divergenza in una di quattro cause:

    modifica_prompt    -> le istruzioni sono ambigue o sbagliate: si puo' correggere
    rilievo_estrazione -> il testo non e' mai arrivato al modello (PDF scansionato,
                          colonne interlacciate): nessun prompt lo risolve
    rilievo_benchmark  -> l'evidenza nel decreto contraddice il file di Alessandro:
                          va portato a lei, NON corretto di nostra iniziativa
    rilievo_capacita   -> sbagliano i modelli piccoli e non i grandi, sulla stessa
                          evidenza: e' un limite di capacita', non di istruzioni

Solo la prima causa autorizza a toccare il prompt. Senza questa separazione il giudice
finirebbe per travestire da regola un bug di estrazione o un errore del benchmark,
peggiorando un prompt che oggi vale F1 0,97.

AUTORITA'. Il benchmark di Alessandro vince per default. Il giudice puo' concludere che
ha ragione il modello solo citando il testo sorgente che contraddice il benchmark, e in
quel caso produce un rilievo_benchmark: una segnalazione per un umano, mai una modifica.

I VINCOLI SONO VALIDATI, NON RACCOMANDATI. Tutto cio' che segue e' verificato da
valida_proposta() prima che la proposta arrivi sotto gli occhi di chi approva:
sezioni protette intatte, al massimo 3 modifiche, nessun riferimento letterale a CUP /
decreti / anni / importi (anti-overfitting), lunghezza entro una banda asimmetrica, e
coerenza fra la lista di modifiche e il prompt materializzato.
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- vincoli
MODELLO_GIUDICE_DEFAULT = "anthropic/claude-opus-5.5"
MAX_MODIFICHE = 3
BANDA_LUNGHEZZA = (0.70, 1.10)      # accorciare si', allungare quasi mai: i modelli
                                    # piccoli peggiorano con i prompt lunghi (e' successo
                                    # a deepseek col passaggio al prompt v2, 2,4x piu' lungo)
SEZIONI_PROTETTE = ("4",)           # 4. FORMATO DI OUTPUT: lo schema a 19 campi. Cambiarlo
                                    # rompe explode_records() e ogni confronto storico.
REGOLE_PROTETTE = ("R6",)           # non dividere l'importo aggregato: la divisione e' una
                                    # convenzione dello SCORER, non del modello.

# tetti del contesto: un PDF da 25 pagine non deve mai entrare intero
CHAR_PER_CUP, CHAR_PER_DOC, CHAR_TOTALE = 3000, 20000, 60000
FINESTRA = 1500

# Struttura di un CUP: lettera + 2 cifre + lettera + 11 cifre = 15 caratteri
# (es. H49J21005480003). La seconda alternativa prende i codici provvisori PROV....
# La terza e' una rete di sicurezza per varianti non standard: 15 caratteri maiuscoli
# con almeno 8 cifre, combinazione che in prosa italiana non capita.
CUP_RE = re.compile(r"\b(?:[A-Z]\d{2}[A-Z]\d{11}|PROV\d{10}|(?=[A-Z0-9]{15}\b)(?:[A-Z0-9]*\d){8}[A-Z0-9]*)\b")
ASSENTE = "CUP ASSENTE DAL TESTO ESTRATTO"


# --------------------------------------------------------------------------- prompt
def estrai_sezione(system: str, numero: str) -> str:
    """Testo della sezione numerata (es. '4' -> '4. FORMATO DI OUTPUT ... fino alla 5)."""
    righe = system.split("\n")
    inizio = fine = None
    for i, r in enumerate(righe):
        if re.match(rf"^{re.escape(numero)}\.\s+[A-ZÀ-Ù]", r.strip()):
            inizio = i
        elif inizio is not None and re.match(r"^\d+\.\s+[A-ZÀ-Ù]", r.strip()):
            fine = i
            break
    if inizio is None:
        return ""
    return "\n".join(righe[inizio:fine]).strip()


def estrai_regola(system: str, sigla: str) -> str:
    """Testo di una regola RN, fino alla regola successiva."""
    righe = system.split("\n")
    inizio = fine = None
    for i, r in enumerate(righe):
        if re.match(rf"^{sigla}\b", r.strip()):
            inizio = i
        elif inizio is not None and re.match(r"^R\d+\b", r.strip()):
            fine = i
            break
    if inizio is None:
        return ""
    return "\n".join(righe[inizio:fine]).strip()


# --------------------------------------------------------------------------- contesto
def finestre_evidenza(testo: str, cup: str) -> list[str]:
    """Ritagli di testo attorno a ogni occorrenza del CUP, piu' il blocco [TABELLA] che lo
    contiene. E' QUI che si tiene piccolo il contesto: si manda il punto contestato, non
    il documento."""
    fuori = []
    for m in re.finditer(re.escape(cup), testo, re.I):
        a, b = max(0, m.start() - FINESTRA), min(len(testo), m.end() + FINESTRA)
        # se il CUP sta dentro un blocco tabella, allarga ai suoi confini: i delimitatori
        # [TABELLA pag.N n.M] rendono il ritaglio pulito e leggibile
        blocco = testo.rfind("[TABELLA", 0, m.start())
        if blocco != -1:
            succ = testo.find("[TABELLA", m.end())
            if blocco >= a - 4000:
                a = min(a, blocco)
                if succ != -1:
                    b = max(b, min(succ, m.end() + 4000))
        fuori.append(testo[a:b].strip()[:CHAR_PER_CUP])
        if sum(len(x) for x in fuori) > CHAR_PER_DOC:
            break
    return fuori or [ASSENTE]


def _ordine(x: tuple) -> tuple:
    return (x[0], x[1], x[2] or "", x[3] or "")


def _riga(x: tuple) -> dict:
    """Riga FN/FP di punteggio.py: (CUP, importo, capitolo/PG, tipologia). Gli ultimi due
    sono None sui documenti valutati al livello base, e allora non si mostrano."""
    cup, imp, pg, tip = x
    d = {"cup": cup, "importo": imp}
    if pg is not None or tip is not None:
        d.update({"capitolo_pg": pg, "tipologia": tip})
    return d


def costruisci_contesto(dettaglio_per_cella: dict, run_dir: Path, prompt: dict,
                        storia: list[dict], note_rifiuto: list[str]) -> dict:
    """Assembla quello che il giudice vede. L'ordine e' anche l'ordine di priorita'."""
    import mcex

    # 1) quali documenti falliscono, e per quali celle
    falliti: dict[str, dict] = defaultdict(lambda: {"celle": [], "fn": set(), "fp": set()})
    for cella, dett in dettaglio_per_cella.items():
        for r in dett:
            if r["solved"]:
                continue
            f = falliti[r["doc"]]
            f["celle"].append(cella)
            f["fn"].update(tuple(x) for x in r["fn_righe"])
            f["fp"].update(tuple(x) for x in r["fp_righe"])

    # 2) evidenza dal PDF, solo per i CUP contestati
    speso = 0
    blocchi = []
    for doc, f in sorted(falliti.items(), key=lambda kv: -len(kv[1]["celle"])):
        pdf = next((p for p in mcex.find_input_pdfs() if p.stem == doc), None)
        if pdf is None:
            continue
        try:
            testo, _, _ = mcex.extract_pdf_text(pdf)
        except Exception as e:
            blocchi.append({"documento": doc, "errore_estrazione": str(e)})
            continue
        cups = {x[0] for x in f["fn"]} | {x[0] for x in f["fp"]}
        ev = {}
        for cup in sorted(cups):
            if speso >= CHAR_TOTALE:
                break
            w = finestre_evidenza(testo, cup)
            ev[cup] = w
            speso += sum(len(x) for x in w)
        blocchi.append({
            "documento": doc,
            "celle_che_falliscono": sorted(set(f["celle"])),
            "mancanti_secondo_alessandro": [_riga(x) for x in sorted(f["fn"], key=_ordine)],
            "in_piu_secondo_il_modello": [_riga(x) for x in sorted(f["fp"], key=_ordine)],
            "evidenza_dal_decreto": ev,
            "record_prodotti": record_contestati(run_dir, doc, cups),
        })

    return {
        "prompt_corrente": {"system": prompt["system"], "user_template": prompt["user_template"]},
        "documenti_falliti": blocchi,
        "storia_round": storia,
        "note_di_rifiuto_precedenti": note_rifiuto,
        "sezioni_protette": {
            "sezione_4_formato_output": "non modificabile",
            "R6_non_dividere_importi_aggregati": "non modificabile",
        },
    }


def record_contestati(run_dir: Path, doc: str, cups: set) -> list[dict]:
    """I record che i modelli hanno prodotto per i CUP contestati, con i campi diagnostici.
    Le regole R13-R14 sono state scritte per far 'mostrare il lavoro' al modello: e' qui
    che ripagano, perche' fanno vedere QUALE COLONNA ha letto rispetto ad Alessandro."""
    out = []
    for f in sorted(run_dir.glob("*/*__think-off.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("decree_file", "")[:-4] != doc:
            continue
        for r in (d.get("records") or []):
            if not (set(r.get("cup") or []) & cups):
                continue
            out.append({
                "modello": d["model_slug"],
                "cup": r.get("cup"), "importo_eur": r.get("importo_eur"),
                "livello_importo": r.get("livello_importo"), "stato_importo": r.get("stato_importo"),
                "annualita": r.get("annualita"), "colonna_fonte": r.get("colonna_fonte"),
                "riferimento_fonte": r.get("riferimento_fonte"),
                "passaggio_testuale": (r.get("passaggio_testuale") or "")[:200],
                "confidenza": r.get("confidenza"), "ambiguita": r.get("ambiguita"),
            })
    return out[:60]


# --------------------------------------------------------------------------- istruzioni
SYSTEM_GIUDICE = """
Sei un revisore metodologico indipendente. Non estrai dati: valuti PERCHE' l'estrazione
automatica di alcuni decreti italiani diverge da un benchmark compilato a mano da
un'analista esperta (Alessandro), e proponi correzioni MIRATE alle istruzioni date ai
modelli estrattori.

AUTORITA'. Il benchmark di Alessandro e' il riferimento. Se modello e benchmark
divergono, l'ipotesi di partenza e' che sbagli il modello. Puoi concludere che ha ragione
il modello SOLO citando testualmente il passo del decreto che contraddice il benchmark: in
quel caso pero' NON proponi una modifica al prompt, produci un rilievo_benchmark da
sottoporre a un umano.

CLASSIFICA OGNI DIVERGENZA IN UNA SOLA CAUSA:
- modifica_prompt: le istruzioni sono ambigue, incomplete o sbagliate. Unica causa che
  autorizza a toccare il prompt.
- rilievo_estrazione: il testo non e' mai arrivato al modello. Il marcatore
  "CUP ASSENTE DAL TESTO ESTRATTO" ne e' la prova diretta.
- rilievo_benchmark: l'evidenza contraddice il benchmark.
- rilievo_capacita: sbagliano i modelli piccoli e non i grandi, sulla stessa evidenza.
  Non e' un difetto delle istruzioni: e' il dato che risponde alla domanda di ricerca.

REGOLE PER LE MODIFICHE AL PROMPT (sono verificate automaticamente: una violazione fa
scartare l'intera proposta prima che un umano la legga):
1. Massimo 3 modifiche. Se ne servissero di piu', scegli le 3 che risolvono piu' documenti.
2. MAI riferimenti letterali: nessun codice CUP, numero di decreto, anno a quattro cifre o
   importo in euro dentro il testo di una regola. Le regole devono valere sul prossimo
   decreto, non su questi. E' il vincolo anti-overfitting.
3. Non toccare la sezione 4 (FORMATO DI OUTPUT) ne' la regola R6. R6 vieta di dividere un
   importo aggregato fra piu' CUP: la divisione e' una convenzione di CHI MISURA, non del
   modello, e replicarla nel prompt corromperebbe il dato.
4. Lunghezza del system prompt fra 0,70x e 1,10x l'attuale. Accorciare e' sempre bene
   accetto: i modelli piccoli peggiorano con prompt lunghi. RIMUOVERE una regola inutile e'
   una modifica legittima e apprezzata.
5. Un documento la cui causa NON e' modifica_prompt non puo' comparire fra i
   documenti_attesi_in_miglioramento.

Restituisci UN SOLO oggetto JSON valido, senza prosa attorno, con questa forma:
{
  "analisi": [{"documento","cup","celle_che_sbagliano":[],"valore_alessandro","valore_modello",
               "evidenza_citata","diagnosi","chi_ha_ragione","azione"}],
  "modifiche_proposte": [{"tipo","target","testo_attuale","testo_nuovo","motivazione",
               "documenti_attesi_in_miglioramento":[],"documenti_a_rischio_regressione":[]}],
  "rilievi_non_prompt": [{"tipo","documento","descrizione"}],
  "prompt_proposto": {"system","user_template"}
}
"testo_attuale" deve comparire alla lettera nel prompt attuale e sparire da quello
proposto; "testo_nuovo" deve comparire alla lettera in "prompt_proposto.system".
"chi_ha_ragione" e' uno fra: alessandro | modello | ambiguo.
"azione" e' uno fra: modifica_prompt | rilievo_estrazione | rilievo_benchmark | rilievo_capacita.
""".strip()


def chiama_giudice(contesto: dict, modello: str, log: Path | None = None) -> dict:
    import mcex
    spec = mcex.ModelSpec(modello, "anthropic", "commercial", "flagship", True, "optional", 4.0, 20.0)
    messaggi = [
        {"role": "system", "content": SYSTEM_GIUDICE},
        {"role": "user", "content":
            "Ecco il prompt attuale, i documenti che divergono dal benchmark, l'evidenza "
            "testuale dei punti contestati e la storia dei round precedenti.\n\n"
            + json.dumps(contesto, ensure_ascii=False, indent=1)},
    ]
    out = mcex.call_openrouter(spec, messaggi, thinking_on=True, log_dir=log, doc_stem="giudice")
    obj, riparato = mcex.parse_model_json(out["content"])
    if obj is None:
        raise RuntimeError("il giudice non ha prodotto JSON valido; "
                           f"finish_reason={out.get('finish_reason')}, "
                           f"primi 300 caratteri: {out['content'][:300]!r}")
    obj["_meta"] = {
        "modello": modello, "json_riparato": riparato,
        "costo_usd": out.get("cost_usd_api"), "token": out.get("total_tokens"),
        "finish_reason": out.get("finish_reason"),
        "latenza_s": out.get("latency_seconds"),
    }
    return obj


# --------------------------------------------------------------------------- validazione
def valida_proposta(p: dict, prompt_corrente: dict) -> list[str]:
    """Le regole scritte in prosa sono suggerimenti; queste sono regole. Restituisce la
    lista dei problemi: se non e' vuota, la proposta non arriva nemmeno all'approvazione."""
    errori: list[str] = []
    sys_now = prompt_corrente["system"]
    prop = p.get("prompt_proposto") or {}
    sys_new, ut_new = prop.get("system"), prop.get("user_template")

    if not isinstance(sys_new, str) or not isinstance(ut_new, str):
        return ["prompt_proposto incompleto: servono 'system' e 'user_template'"]

    # segnaposto del decreto
    if ut_new.count("<<DECREE_TEXT>>") != 1:
        errori.append(f"user_template contiene {ut_new.count('<<DECREE_TEXT>>')} segnaposto "
                      f"<<DECREE_TEXT>> invece di 1")

    # banda di lunghezza
    r = len(sys_new) / max(len(sys_now), 1)
    if not (BANDA_LUNGHEZZA[0] <= r <= BANDA_LUNGHEZZA[1]):
        errori.append(f"lunghezza del system prompt {r:.2f}x l'attuale, fuori dalla banda "
                      f"{BANDA_LUNGHEZZA[0]}-{BANDA_LUNGHEZZA[1]}x")

    # sezioni e regole protette, byte per byte
    for n in SEZIONI_PROTETTE:
        if estrai_sezione(sys_now, n).strip() != estrai_sezione(sys_new, n).strip():
            errori.append(f"la sezione protetta {n} (FORMATO DI OUTPUT) e' stata modificata")
    for sigla in REGOLE_PROTETTE:
        if estrai_regola(sys_now, sigla).strip() != estrai_regola(sys_new, sigla).strip():
            errori.append(f"la regola protetta {sigla} e' stata modificata")

    mods = p.get("modifiche_proposte") or []
    if len(mods) > MAX_MODIFICHE:
        errori.append(f"{len(mods)} modifiche proposte, il massimo e' {MAX_MODIFICHE}")

    # anti-overfitting + coerenza lista/prompt materializzato
    for i, m in enumerate(mods, 1):
        nuovo = m.get("testo_nuovo") or ""
        attuale = m.get("testo_attuale") or ""
        # Gli ESEMPI della sezione 5 sono casistici per natura: contengono CUP e importi
        # veri, ed e' giusto cosi'. Sono pero' esentati dall'anti-overfitting SOLO se si
        # tratta di un'aggiunta: un esempio esistente non si modifica, perche' cambiarlo
        # significa riscrivere un caso guida su cui il prompt e' gia' tarato.
        e_esempio = (m.get("tipo") == "modifica_esempio"
                     or str(m.get("target", "")).lower().startswith("esempio"))
        if e_esempio and attuale.strip():
            errori.append(f"modifica {i} ({m.get('target')}): un esempio della sezione 5 puo' "
                          f"essere AGGIUNTO, non modificato")
        # Un esempio guida E' un caso concreto: CUP, importi e annualita' veri sono la sua
        # sostanza, non un difetto. Resta vincolato all'aggiunta (sopra), che e' cio' che
        # impedisce di ritagliarlo sui documenti che stiamo inseguendo.
        controlli = () if e_esempio else (
                    ("un codice CUP", CUP_RE),
                    ("un anno a 4 cifre", re.compile(r"\b(?:19|20)\d{2}\b")),
                    ("un importo in euro", re.compile(r"\d{1,3}(?:\.\d{3}){2,}")),
                    ("un numero di decreto", re.compile(r"\bn\.\s?\d{2,4}\b")))
        for etichetta, rx in controlli:
            if rx.search(nuovo):
                errori.append(f"modifica {i} ({m.get('target')}): il testo nuovo contiene "
                              f"{etichetta} — le regole devono valere sul prossimo decreto")
        if nuovo and nuovo not in sys_new:
            errori.append(f"modifica {i} ({m.get('target')}): 'testo_nuovo' non compare nel "
                          f"prompt proposto — diff dichiarato ma non applicato")
        if attuale:
            if attuale not in sys_now:
                errori.append(f"modifica {i} ({m.get('target')}): 'testo_attuale' non si trova "
                              f"nel prompt attuale")
            elif attuale in sys_new:
                errori.append(f"modifica {i} ({m.get('target')}): 'testo_attuale' e' ancora "
                              f"presente nel prompt proposto")

    # regola di autorita': una causa non-prompt non puo' promettere miglioramenti
    non_prompt = {a.get("documento") for a in (p.get("analisi") or [])
                  if a.get("azione") and a["azione"] != "modifica_prompt"}
    prompt_doc = {a.get("documento") for a in (p.get("analisi") or [])
                  if a.get("azione") == "modifica_prompt"}
    for i, m in enumerate(mods, 1):
        for d in (m.get("documenti_attesi_in_miglioramento") or []):
            if d in non_prompt and d not in prompt_doc:
                errori.append(f"modifica {i}: promette di migliorare '{d}', che pero' e' stato "
                              f"classificato come causa non risolvibile dal prompt")
    return errori


# --------------------------------------------------------------------------- relazione
def scrivi_relazione(p: dict, errori: list[str], vinti_persi: str, dest: Path) -> None:
    L = ["# Proposta del revisore — round da approvare", ""]
    if vinti_persi:
        L += ["## Variazioni rispetto al round precedente", "", vinti_persi, ""]
    if errori:
        L += ["## PROPOSTA SCARTATA AUTOMATICAMENTE", "",
              "Ha violato i vincoli verificati dall'orchestratore:", ""]
        L += [f"- {e}" for e in errori] + [""]
    mods = p.get("modifiche_proposte") or []
    L += [f"## Modifiche proposte ({len(mods)})", ""]
    for i, m in enumerate(mods, 1):
        L += [f"### {i}. {m.get('tipo')} su {m.get('target')}", "",
              f"**Perche':** {m.get('motivazione')}", "",
              f"**Documenti attesi in miglioramento:** {', '.join(m.get('documenti_attesi_in_miglioramento') or []) or '—'}", "",
              f"**A rischio di regressione:** {', '.join(m.get('documenti_a_rischio_regressione') or []) or '—'}", "",
              "```diff", "- " + (m.get("testo_attuale") or "(nuova regola)").replace("\n", "\n- "),
              "+ " + (m.get("testo_nuovo") or "(rimossa)").replace("\n", "\n+ "), "```", ""]
    an = p.get("analisi") or []
    if an:
        L += ["## Analisi per documento", "",
              "| documento | CUP | Alessandro | modello | diagnosi | ha ragione | causa |",
              "|---|---|---|---|---|---|---|"]
        for a in an:
            L.append(f"| {a.get('documento','')} | {a.get('cup','')} | {a.get('valore_alessandro','')} | "
                     f"{a.get('valore_modello','')} | {a.get('diagnosi','')} | "
                     f"{a.get('chi_ha_ragione','')} | **{a.get('azione','')}** |")
        L.append("")
    ril = p.get("rilievi_non_prompt") or []
    if ril:
        L += ["## Rilievi che NON si risolvono col prompt", ""]
        for r in ril:
            L.append(f"- **[{r.get('tipo')}]** {r.get('documento')}: {r.get('descrizione')}")
        L.append("")
    dest.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Revisore terzo del ciclo di prompt.")
    ap.add_argument("--contesto", type=Path, required=True, help="JSON del contesto")
    ap.add_argument("--prompt", type=Path, required=True, help="prompt attuale (vNNN.json)")
    ap.add_argument("--out", type=Path, required=True, help="dove scrivere proposta.json")
    ap.add_argument("--modello", type=str, default=MODELLO_GIUDICE_DEFAULT)
    args = ap.parse_args()

    contesto = json.loads(args.contesto.read_text(encoding="utf-8"))
    prompt = json.loads(args.prompt.read_text(encoding="utf-8"))
    p = chiama_giudice(contesto, args.modello)
    errori = valida_proposta(p, prompt)
    p["_validazione"] = {"ok": not errori, "errori": errori}
    args.out.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"proposta -> {args.out}   validazione: {'OK' if not errori else 'SCARTATA'}")
    for e in errori:
        print(f"  - {e}")


if __name__ == "__main__":
    main()
