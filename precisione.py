"""Precisione sugli atti senza CUP (Q5): qualunque CUP trovato li' e' un falso positivo.
Uso: python precisione.py <cartella_run>
"""
import sys
from pathlib import Path

import pandas as pd

el = {l.strip() for l in open("documenti_eleggibili.txt", encoding="utf-8")
      if l.strip() and not l.startswith("#")}
r = pd.read_csv(Path(sys.argv[1]) / "records.csv")
stem = r["decree_file"].str.removesuffix(".pdf")
fp = r[~stem.isin(el) & ~stem.str.contains("delibera CIPE") & r["cup"].notna()]
print(fp.groupby(["model_slug", "decree_file"]).size() if len(fp) else "nessun CUP inventato")
