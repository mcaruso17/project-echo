#!/usr/bin/env python3
"""Conteggio token per file di ./input (o della cartella passata come argomento).

Flusso: 1) modello; 2) solo modelli locali, se ci sono PDF o immagini: testo, immagini o entrambi;
3) se le pagine PDF diventano immagini: DPI (default 108, quello della web UI di llama.cpp).

Testo
  OpenAI    tiktoken o200k_base                   https://github.com/openai/tiktoken (tiktoken/model.py)
  DeepSeek  tokenizer.json, BPE byte-level 128k   https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
  Kimi      tiktoken.model + regex KIMI_PAT       https://huggingface.co/moonshotai/Kimi-K3 (tokenization_kimi.py)
  GLM       tokenizer.json, BPE byte-level 155k   https://huggingface.co/zai-org/GLM-5.3
  Senza token speciali né template di chat.

Immagini, API OpenAI, detail "high"   https://developers.openai.com/api/docs/guides/images-vision
  tile   fit 2048x2048, lato corto 768, tile 512px: base + tile*n (85/170 GPT-4o/4.1, 70/140 GPT-5/5.1)
  patch  lato lungo 2048, patch 32px, budget 2500 (shrink_factor): ceil(patch*1.2) (GPT-5.4/5.5/5.6)

Immagini, modelli locali: preprocessing di llama.cpp con i parametri di default del mmproj
  https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd (mtmd-image.cpp, clip.cpp, mtmd.cpp)
  DeepSeek V4.1-Flash  deepseek41v  patch 14, merge 3, blocco h*(w+1)+2 <= 1024 token, min_pixels 295936
  Kimi K2.5 / K2.6     kimik25      lati arrotondati a 28px, 8..16384 patch da 14px, +2 marcatori
  GLM-5.3-Flash        glm5v        canvas allineato per eccesso a 28px, 16..8000 token, +2 marcatori

PDF
  API OpenAI: testo + un'immagine per pagina a OPENAI_PDF_DPI
    https://developers.openai.com/api/docs/guides/pdf-files
  llama.cpp web UI: solo testo (default) oppure solo immagini ("Parse PDF as image"), PDF.js a scala 1.5
    https://github.com/ggml-org/llama.cpp/blob/master/tools/ui/src/lib/utils/convert-files-to-extra.ts

Flowchart delle opzioni, verifiche e contesto: README.md
"""
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple

# Se lanciato con il Python di sistema, si rilancia con quello del .venv (dove ci sono le librerie)
VENV_PY = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
if sys.prefix == sys.base_prefix and VENV_PY.exists():
    os.execv(VENV_PY, [str(VENV_PY), *sys.argv])

import tiktoken
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import logging as hf_logging
from PIL import Image
from pypdf import PdfReader
from tiktoken.load import load_tiktoken_bpe
from tokenizers import Tokenizer

INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "input"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
OPENAI_PDF_DPI = 150  # ipotesi: OpenAI non documenta la risoluzione delle pagine
LLAMACPP_PDF_DPI = 72 * 1.5  # default dei modelli locali: web UI llama.cpp, PDF.js getViewport({scale: 1.5})
COL = 14  # larghezza di una colonna della tabella
hf_logging.set_verbosity_error()  # i repo dei tokenizer sono pubblici: niente avviso "unauthenticated requests"
# Avvisi di pypdf sui PDF imperfetti ("fontTools is required", "Ignoring wrong pointing object"):
# li aggira da solo e il testo estratto non cambia (verificato con fontTools installato)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Regex di pre-tokenizzazione di Kimi, copiata da tokenization_kimi.py (uguale da K2 a K3)
KIMI_PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


def openai_encoder():
    enc = tiktoken.get_encoding("o200k_base")
    return lambda text: len(enc.encode(text, disallowed_special=()))


def hf_encoder(repo):
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def kimi_encoder(repo):
    ranks = load_tiktoken_bpe(hf_hub_download(repo, "tiktoken.model"))
    enc = tiktoken.Encoding("kimi", pat_str=KIMI_PAT, mergeable_ranks=ranks, special_tokens={})
    return lambda text: len(enc.encode(text, disallowed_special=()))


def tile_tokens(w, h, base, tile):
    # Entra in 2048x2048, poi lato corto a 768px, poi conta i tile da 512px
    scale = min(1, 2048 / max(w, h))
    w, h = int(w * scale), int(h * scale)
    scale = min(1, 768 / min(w, h))
    w, h = int(w * scale), int(h * scale)
    return base + tile * math.ceil(w / 512) * math.ceil(h / 512)


def patch_tokens(w, h, mult, max_side=2048, budget=2500):
    # Lato lungo entro max_side, poi se le patch da 32px superano il budget rimpicciolisce
    # (formula shrink_factor della documentazione OpenAI)
    scale = min(1, max_side / max(w, h))
    w, h = w * scale, h * scale
    if math.ceil(w / 32) * math.ceil(h / 32) > budget:
        shrink = math.sqrt(32 * 32 * budget / (w * h))
        shrink *= min(math.floor(w * shrink / 32) / (w * shrink / 32),
                      math.floor(h * shrink / 32) / (h * shrink / 32))
        w, h = w * shrink, h * shrink
    w, h = int(w), int(h)
    return math.ceil(math.ceil(w / 32) * math.ceil(h / 32) * mult)


def deepseek41_tokens(w, h, p=14, r=3, max_tok=1024, min_px=295936):
    # mtmd_image_preprocessor_deepseek41v::plan_grid; il blocco include start, end e un newline per riga
    def n_tok(bw, bh):
        return -(-(bh // p) // r) * (-(-(bw // p) // r) + 1) + 2
    if w * h < min_px:
        ratio = math.sqrt(min_px / (w * h))
        w, h = int(w * ratio), int(h * ratio)
    bw, bh = -(-w // p) * p, -(-h // p) * p
    if n_tok(bw, bh) > max_tok:
        cell = p * r
        max_w = math.sqrt((max_tok - 2) / (h / w) + 0.25) - 0.5
        max_h = max_w * h / w
        if max_w < 1:
            bw, bh = cell, (max_tok - 2) // 2 * cell
        elif max_h < 1:
            bw, bh = (max_tok - 3) * cell, cell
        else:
            beta = min(math.floor(max_w) * cell / w, math.floor(max_h) * cell / h)
            bw, bh = math.floor(w * beta / p) * p, math.floor(h * beta / p) * p
    return n_tok(bw, bh)


def kimik25_tokens(w, h, f=28, min_px=8 * 14 * 14, max_px=16384 * 14 * 14):
    # mtmd_image_preprocessor_dyn_size (calc_size_preserved_ratio), + <|media_begin|> <|media_end|>
    bw, bh = max(f, math.floor(w / f + 0.5) * f), max(f, math.floor(h / f + 0.5) * f)
    if bw * bh > max_px:
        beta = math.sqrt(w * h / max_px)
        bw, bh = max(f, math.floor(w / beta / f) * f), max(f, math.floor(h / beta / f) * f)
    elif bw * bh < min_px:
        beta = math.sqrt(min_px / (w * h))
        bw, bh = math.ceil(w * beta / f) * f, math.ceil(h * beta / f) * f
    return bw // f * (bh // f) + 2


def glm5v_tokens(w, h, f=28, min_px=16 * 28 * 28, max_px=8000 * 28 * 28):
    # mtmd_image_preprocessor_glm5v, + <|begin_of_image|> <|end_of_image|>
    def align(v):
        return -(-v // f) * f
    cw, ch = align(w), align(h)
    if cw * ch < min_px:
        scale = math.sqrt(min_px / (w * h))
        cw, ch = align(max(1, math.ceil(w * scale))), align(max(1, math.ceil(h * scale)))
    if cw * ch > max_px:
        # altezza massima del contenuto il cui canvas allineato sta nel budget
        lo, hi, cw, ch = 1, h, f, f
        while lo <= hi:
            mid = (lo + hi) // 2
            aw, ah = align(max(1, w * mid // h)), align(mid)
            if aw * ah <= max_px:
                cw, ch, lo = aw, ah, mid + 1
            else:
                hi = mid - 1
    return cw // f * (ch // f) + 2


class Modello(NamedTuple):
    colonna: str  # intestazione nella tabella
    voce: str  # voce del menu
    locale: bool  # False = API OpenAI, i PDF arrivano come testo + immagini
    tokenizer: Callable
    immagine: Callable | None  # token per un'immagine w x h; None = senza visione


MODELLI = [
    Modello("GPT-5.4+", "GPT-5.4 / GPT-5.5 / GPT-5.6", False, openai_encoder, partial(patch_tokens, mult=1.2)),
    Modello("GPT-5/5.1", "GPT-5 / GPT-5.1", False, openai_encoder, partial(tile_tokens, base=70, tile=140)),
    Modello("GPT-4o/4.1", "GPT-4o / GPT-4.1", False, openai_encoder, partial(tile_tokens, base=85, tile=170)),
    Modello("DeepSeek", "DeepSeek V3.2 / V4 / V4.1", True,
            partial(hf_encoder, "deepseek-ai/DeepSeek-V4.1-Flash"), deepseek41_tokens),
    Modello("Kimi", "Kimi K2 / K2.5 / K2.6 / K3", True, partial(kimi_encoder, "moonshotai/Kimi-K3"), kimik25_tokens),
    Modello("GLM", "GLM-4.7 / GLM-5 / GLM-5.1 / GLM-5.3", True, partial(hf_encoder, "zai-org/GLM-5.3"), glm5v_tokens),
]
VARIANTI_VISIONE = "DeepSeek V4.1-Flash, Kimi K2.5 / K2.6, GLM-5.3-Flash"
MODI = {"1": ["testo"], "2": ["immagini"], "3": ["testo", "immagini"]}


def chiedi(domanda):
    try:
        return input(domanda).strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nAnnullato")


def scegli_modelli():
    print("Per quale modello contare i token?")
    for i, m in enumerate(MODELLI, 1):
        if i == 1 or m.locale != MODELLI[i - 2].locale:
            print("  Locale (llama.cpp)" if m.locale else "  API OpenAI")
        print(f"    {i}) {m.voce}")
    print("  t) tutti, per confrontarli")
    while True:
        risposta = chiedi("Scelta (es. 1 oppure 1,4,5) [1]: ") or "1"
        if risposta == "t":
            return MODELLI
        numeri = risposta.replace(",", " ").split()
        if all(n.isdigit() and 1 <= int(n) <= len(MODELLI) for n in numeri):
            return [MODELLI[int(n) - 1] for n in dict.fromkeys(numeri)]
        print("Scelta non valida")


def scegli_modi():
    print("\nCome arrivano PDF e immagini al modello locale?")
    print("  1) testo: dei PDF solo il testo estratto, immagini non lette (default di llama.cpp)")
    print("  2) immagini: ogni pagina del PDF diventa un'immagine, senza testo (\"Parse PDF as image\")")
    print(f"     serve la variante con visione: {VARIANTI_VISIONE}")
    print("     ogni pagina costa un numero fisso di token, che dipende dai DPI e non dal testo")
    print("  3) entrambi, per confrontarli")
    while True:
        risposta = chiedi("Scelta [3]: ") or "3"
        if risposta in MODI:
            return MODI[risposta]
        print("Scelta non valida")


def massimo_a4(image_tokens):
    """(token massimi di una pagina A4, primo DPI che li raggiunge): oltre, i DPI non aggiungono token."""
    valori = [(image_tokens(int(595.28 * d / 72), int(841.89 * d / 72)), d) for d in range(50, 601)]
    massimo = max(v for v, _ in valori)
    return massimo, next(d for v, d in valori if v == massimo)


def scegli_dpi(modelli):
    print("\nA quanti DPI le pagine PDF diventano immagini?")
    print(f"  {LLAMACPP_PDF_DPI:g} = valore fisso della web UI di llama.cpp, non uno standard."
          " Se converti tu le pagine, i DPI che usi.")
    print("  Oltre una soglia il modello rimpicciolisce la pagina: i DPI in più non aggiungono token (pagina A4):")
    for m in modelli:
        if m.locale and m.immagine:
            massimo, dpi = massimo_a4(m.immagine)
            print(f"    {m.colonna:<10} massimo {massimo:,} token, da {dpi} DPI")
    while True:
        risposta = chiedi(f"DPI [{LLAMACPP_PDF_DPI:g}]: ") or f"{LLAMACPP_PDF_DPI:g}"
        try:
            dpi = float(risposta.replace(",", "."))
        except ValueError:
            dpi = 0
        if dpi > 0:
            return dpi
        print("Scelta non valida")


def colonne(modelli, modi):
    """(intestazione, modello, modo) per ogni colonna; modo None = API OpenAI."""
    cols = []
    for m in modelli:
        if not m.locale:
            cols.append((m.colonna, m, None))
        else:
            cols += [(m.colonna + (" img" if modo == "immagini" else ""), m, modo) for modo in modi]
    return cols


def leggi(path):
    """Restituisce (testi, immagini in pixel, pagine PDF in punti) contenuti nel file."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = PdfReader(path).pages
        sizes = []
        for p in pages:
            w, h = float(p.cropbox.width), float(p.cropbox.height)
            sizes.append((h, w) if p.rotation % 180 else (w, h))
        return [p.extract_text() or "" for p in pages], [], sizes
    if ext in IMAGE_EXT:
        with Image.open(path) as img:
            return [], [img.size], []
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        raise ValueError("formato binario non supportato")
    return [data.decode("utf-8", errors="ignore")], [], []


def conta(testi, immagini, pagine, encoder, modello, modo, dpi_locale):
    """Token del file per una colonna; None se il file è solo immagini e la colonna non le legge."""
    if modo == "testo":
        return None if immagini else sum(encoder(t) for t in testi)
    dpi = OPENAI_PDF_DPI if modo is None else dpi_locale
    n = sum(modello.immagine(w, h) for w, h in immagini)
    n += sum(modello.immagine(int(w * dpi / 72), int(h * dpi / 72)) for w, h in pagine)
    if modo is None or not pagine:  # in llama.cpp un PDF come immagini non porta il testo
        n += sum(encoder(t) for t in testi)
    return n


def main():
    files = sorted(p for p in INPUT.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not files:
        sys.exit(f"Nessun file in {INPUT}")
    modelli = scegli_modelli()
    visivi = any(f.suffix.lower() == ".pdf" or f.suffix.lower() in IMAGE_EXT for f in files)
    modi = scegli_modi() if visivi and any(m.locale and m.immagine for m in modelli) else ["testo"]
    pdf = any(f.suffix.lower() == ".pdf" for f in files)
    dpi = scegli_dpi(modelli) if pdf and "immagini" in modi else LLAMACPP_PDF_DPI
    encoders = {}
    for m in modelli:
        try:
            encoders[m.colonna] = m.tokenizer()
        except Exception as e:
            sys.exit(f"Impossibile caricare il tokenizer {m.voce}: {e}")

    cols = colonne(modelli, modi)
    print("\n" + "".join(f"{h:>{COL}}" for h, _, _ in cols) + "  file")
    totali = [0] * len(cols)
    esclusi = [False] * len(cols)
    counted = 0
    for f in files:
        try:
            letto = leggi(f)
            risultati = [conta(*letto, encoders[m.colonna], m, modo, dpi) for _, m, modo in cols]
        except Exception as e:
            print(f"{'ERRORE':>{COL * len(cols)}}  {f.relative_to(INPUT)}  ({e})")
            continue
        for i, n in enumerate(risultati):
            totali[i] += n or 0
            esclusi[i] |= n is None
        counted += 1
        print("".join(f"{'n/d' if n is None else f'{n:,}':>{COL}}" for n in risultati) + f"  {f.relative_to(INPUT)}")
    print("-" * COL * len(cols))
    print("".join(f"{f'{t:,}' + ('*' if e else ''):>{COL}}" for t, e in zip(totali, esclusi))
          + f"  TOTALE ({counted} file)")
    if any(esclusi):
        print("\nn/d = immagine non letta da un modello usato senza visione; * = totale senza quei file")


if __name__ == "__main__":
    main()
