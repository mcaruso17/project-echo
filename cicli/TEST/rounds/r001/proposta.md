# Proposta del revisore — round da approvare

## Variazioni rispetto al round precedente

| cella | solved | vs round prec. | nuovi | persi |
|---|---|---|---|---|
| z-ai/glm-4.7-flash::off | 1/2 | — | 2025 DM MIT 131-2025 | **—** |

## Modifiche proposte (0)

## Analisi per documento

| documento | CUP | Alessandro | modello | diagnosi | ha ragione | causa |
|---|---|---|---|---|---|---|
| 2021 DM mit 464-2021 | H94I19000130004 | 97154124.0 (livello totale, un record per la riga dell'Allegato 1) | 97154124.1, livello_importo = quota | CUP e importo sono corretti: la differenza di 0,10 euro è solo l'arrotondamento del benchmark. La divergenza nasce, per inferenza, dal campo livello_importo. Il modello ha etichettato la riga come 'quota', probabilmente perché ha letto la riga finale 'TOTALE | 660.660.678,33' come il totale di cui le righe sarebbero le quote. R5 invece definisce totale e quote rispetto allo stesso CUP. Qui ogni CUP ha un solo importo, quindi il record è 'totale'. È così che risulta mancante nel confronto. Le altre celle, con modelli più grandi o con il ragionamento attivo, non falliscono su questo documento con la stessa evidenza e lo stesso prompt. | alessandro | **rilievo_capacita** |
| 2021 DM mit 464-2021 | C81B21013200005 | 359545887.0 (livello totale) | 359545887.07, livello_importo = quota | Stesso errore sistematico: tutte e cinque le righe sono etichettate 'quota' per aver scambiato la riga TOTALE della tabella, che somma CUP diversi, per un totale per CUP. Importo e CUP sono corretti. | alessandro | **rilievo_capacita** |
| 2021 DM mit 464-2021 | J80J18000000001 | 159466174.0 (livello totale) | 159466174.12, livello_importo = quota | Stesso errore di livello_importo. Estrazione di CUP e importo corretta. | alessandro | **rilievo_capacita** |
| 2021 DM mit 464-2021 | C11B21008720001 | 9408503.0 (livello totale) | 9408502.67, livello_importo = quota | Stesso errore di livello_importo. Lo scarto di 0,33 euro è dovuto all'arrotondamento all'unità nel benchmark e non contraddice il testo. | alessandro | **rilievo_capacita** |
| 2021 DM mit 464-2021 | D81E20000410007 | 35085990.0 (livello totale) | 35085990.37, livello_importo = quota | Stesso errore di livello_importo. Estrazione di CUP e importo corretta. | alessandro | **rilievo_capacita** |

## Rilievi che NON si risolvono col prompt

- **[rilievo_capacita]** 2021 DM mit 464-2021: Fallisce una sola cella, glm-4.7-flash con ragionamento disattivato. Le cinque coppie CUP-importo sono estratte correttamente. Tutti i record sono però etichettati livello_importo='quota' invece di 'totale': il modello ha interpretato la riga 'TOTALE' in fondo al piano di riparto, che somma CUP diversi, come il totale di cui le righe sarebbero le quote. R5 lega già totale e quote allo stesso CUP, e i modelli più grandi, sulla stessa evidenza e con lo stesso prompt, non sbagliano. Non è quindi un difetto delle istruzioni ma un limite del modello piccolo senza ragionamento: è il dato che risponde alla domanda di ricerca. Per questo non propongo modifiche al prompt. Se in round futuri lo stesso errore comparisse anche su modelli grandi, andrebbe valutata una precisazione in R5: la riga di totale di colonna non trasforma le righe dei singoli CUP in quote. Nota secondaria per chi misura: il benchmark arrotonda gli importi all'euro, mentre il decreto riporta i centesimi (es. 9.408.502,67 contro 9408503). Il confronto deve tollerare scarti inferiori a un euro; qui non è la causa del fallimento, perché le altre celle passano.
