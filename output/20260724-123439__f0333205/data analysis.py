
import os
import pandas as pd
import json

user = "matteo.caruso"
path = fr"C:\Users\{user}\OneDrive - mef.gov.it\File di Olivieri Giuseppe - AAAA Previsioni di Spesa\5. LLMs\output\20260724-123439__f0333205"
os.chdir(path)

print("Import the results ...")
print("   ")

df = pd.read_json("results.jsonl", lines=True)

print("Results imported!")
print("   ")
print("Extrapolate the nested JSON ...")
records = df.explode("records")

records = pd.concat(
    [
        records.drop(columns="records"),
        pd.json_normalize(records["records"])
    ],
    axis=1
)
print("   ")
print("Records extrapolated!")
print("   ")
print("Summary statistics:")
print("   ")
recshape= records.shape
print(f"It's a dataframe with {recshape[0]} observations")
print(f"and {recshape[1]} columns")
print("  ")
print(f"we have {records["cup"].nunique()} unique CUPs,")
print(f"with {records["cup"].isna().sum()} observations missing it.")
print("  ")
azioni = records["tipo_azione"].value_counts(dropna=False)
print("We have the following financial movements:")
print("  ")
print(f"Assegnazioni: {azioni["assegnazione"]}")
print(f"Rimodulazioni: {azioni["rimodulazione"]}")
print(f"and {records["tipo_azione"].isna().sum()} observations with missing values")
print("  ")
print("The import in EUR ")
print(records["importo_eur"].round(3).describe())
print(records["cup"].str.len().value_counts())


