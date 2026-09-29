# Convenzioni di estrazione — domande per Alessandro

**A cosa serve questa scheda.** Stiamo misurando quanto bene diversi modelli linguistici
riescono a estrarre dai decreti le coppie CUP–importo, usando il tuo lavoro manuale
(`Output a MANO.xlsx`) come riferimento. Il confronto è automatico e a corrispondenza
esatta: per ogni documento l'insieme delle coppie estratte deve coincidere con il tuo,
senza righe mancanti né in più.

**Il risultato di partenza è già buono**: il modello migliore riproduce esattamente il tuo
lavoro su **13 documenti su 16**. Sui 3 che restano, però, quasi nulla è un errore di lettura: sono **casi in cui tu e lui
hanno seguito convenzioni diverse**. In uno il modello aveva trovato esattamente la cifra
giusta e l'ha solo classificata in modo diverso dal tuo; in un altro non sappiamo ancora
chi dei due abbia ragione.

Le convenzioni giuste sono le tue. Il problema è che non sono scritte da nessuna parte, e
quindi non possiamo spiegarle al modello. **Le tue risposte qui sotto diventano le regole
che gli daremo.** Non c'è nessuna risposta "sbagliata": serve sapere come lavori.

Il file `convenzioni_da_chiarire.xlsx` ha un foglio per ogni domanda, con le righe
affiancate e il testo del decreto da cui provengono.

---

## Q1 — Un codice marcato "CUP MASTER" vale come CUP? *(serve solo una conferma)*
*Foglio: `Q1 CUP Master`*

In `2022 DI Mims Mef 97 20.04.2022` il codice nella tabella è scritto così:

> `F41B22000740009 (CUP MASTER)`

Nel tuo file quella riga c'è, con **€ 631.373.163**. Le istruzioni che abbiamo dato ai
modelli dicono invece che un "CUP master" va tenuto **separato** dal CUP della riga — una
regola che, alla prova dei fatti, sembra sbagliata: **tutti e dieci i modelli l'hanno
ignorata** e hanno attribuito l'importo a quel CUP, esattamente come hai fatto tu.

Quindi qui non c'è un disaccordo, c'è una nostra istruzione da correggere. Ci basta la
tua conferma:

- [X] Sì, quando il CUP compare solo come "(CUP MASTER)" è il CUP di quella riga
- [ ] No, attenzione: ................................

---

## Q2 — Una riga con più colonne di importo: quante righe genera?
*Foglio: `Q2 colonne multiple`*

In `2023 DM MIT 342-2023` il CUP `H49J21005480003` compare nel tuo file **nove volte**, con
nove importi diversi (da € 4.025 a € 86.250). Il modello ne ha estratto **uno solo**:
€ 44.275, dalla colonna "IMPORTO AMMESSO A FINANZIAMENTO CENTRO-NORD – POTENZIAMENTO".

Sembra che l'allegato abbia più colonne di importo (Centro-Nord / Sud incrociate con le
tipologie di intervento) e che tu prenda **una riga per ciascuna colonna valorizzata**,
mentre al modello abbiamo detto di scegliere la colonna che l'atto dispone.

**Domanda:** una riga di tabella con più colonne di importo genera...

- [ ] Una riga per ogni colonna valorizzata (è quello che sembra dal tuo file)
- [ ] Una riga sola, per la colonna che l'atto dispone
- [ ] Dipende — da cosa? il problema in questo senso non riguarda il numero di colonne; prendere una sola colonna nella nostra casistica specifica è giusto perché a noi interessa la colonna che fa riferimento all'importo totale (in verde con il titolo "IMPORTO AMMASSO A 
FINANZIAMENTO CENTRO - NORD - POTENZIAMENTO (euro)"). La differenza reale tra il risultato di Alessandro e quello degli LLMs è che Alessandro ha considerato che ci sono diverse righe con più importi per lo stesso CUP. Questo è dovuto al fatto che i CUP possono avere diversi interventi al loro interno. Ogni intervento può avere diversi importi. In questo caso, Alessandro ha correttamente preso più importi, dalla stessa colonna citata prima, ma che fanno riferimento a diversi interventi per lo stesso CUP. Gli LLMs si sono limitati a prendere solo il primo importo (i.e. "44.275,00"). Questo può essere dovuto al fatto che la formattazione crea un unica grande cella con il numero di CUP che però è collegata a più righe per i diversi interventi. Laddove c'è questa casistica, LLM deve riportare più importi, specificando i differenti interventi per ogni importo relativi allo stesso CUP. 

Se vale la prima: l'importo totale della riga va comunque registrato a parte, o solo i
singoli per colonna?

---

## Q3 — Il ponte piano gestionale → CUP: cosa registriamo da un decreto di impegno
*Foglio: `Q3 totale vs quota`* — **è la domanda che pesa di più** *(rivista il 28/09)*

### Perché la domanda è cambiata

La prima versione chiedeva "qual è il totale". La tua risposta (*solo quello
effettivamente impegnato da questo atto, senza quote annuali*) ha spostato il problema
nel posto giusto: non conta **quale cifra è più grande**, conta **in che stato si trovano
le risorse**. Lo stesso denaro, per lo stesso CUP, attraversa più atti:

| Stato | Che cosa dice | Atto tipico | Esempio nel campione |
|---|---|---|---|
| **stanziato** | la legge mette risorse su un capitolo / PG | Legge di bilancio | L. 197/2022, art. 1 c. 478: € 2.200.000.000 |
| **assegnato** | il riparto attribuisce le risorse a un intervento | DM / DI di riparto | DM 464/2021 (graduatoria TRM) |
| **impegnato** | un PG viene impegnato a favore di un CUP, per un importo | DD di impegno | DD 128/2022, DD 454/2022, DD 141/2025 |

Il nostro obiettivo è seguire le risorse **da quando sono stanziate a quando sono
impegnate su un progetto**. L'anello che nessun sistema oggi ricostruisce è l'ultimo:
**quale piano gestionale è impegnato per quale CUP, e per quanto**. È su questo che va
costruita l'unità di registrazione, e il rapporto è molti-a-molti: un CUP può attingere
a più PG, un PG può finanziare più CUP.

### Il caso `2025 DD MIT 141-2025` riletto con questo criterio

Il decreto nomina quattro somme dello stesso rango. Solo due sono **impegni disposti da
questo atto**:

| Dove | Importo | Capitolo / PG | Che cos'è | Da registrare qui? |
|---|---:|---|---|---|
| art. 2 | € 3.945.000.000 | — | valore complessivo della Convenzione (somma delle tre fonti) | no — non è un movimento |
| art. 2, 1° alinea | € 145.000.000 | 7426 / 1 | **già impegnato** dal DD 454/2022 | no — si registra dal DD 454/2022 |
| art. 2, 2° alinea | € 1.600.000.000 | 7400 / 1 e 7400 / 8 | **già impegnato** dal DD 419/2022 | no — si registra dal DD 419/2022 |
| art. 2, 3° alinea | € 2.200.000.000 | 7419 / 2 | risorse **stanziate** dalla L. 197/2022 | no — è lo stanziamento |
| **art. 3** | **€ 2.125.000.000** | **7419 / 2** | **impegno disposto da questo atto** | **sì** |
| **art. 4** | **€ 75.000.000** | **7416 / 1** | **impegno disposto da questo atto** | **sì** |

2.125.000.000 + 75.000.000 = **2.200.000.000**: la cifra nel tuo file è esattamente la
somma dei due impegni. Il dato non è sbagliato, ma **è aggregato a livello di CUP e
perde il ponte**: l'art. 2 attribuisce i 2,2 miliardi interamente al 7419/2, mentre gli
artt. 3-4 ne impegnano 75 milioni su un altro capitolo (7416/1). Con una riga sola quello
spostamento sparisce, e sparisce proprio l'informazione che ci serve per agganciare
INIT / BDAP.

Se registrassimo anche i 145 milioni e l'1,6 miliardi, li conteremmo **due volte**: il
DD 454/2022 è nel campione e tu ne hai già registrato i € 14.000.000 + € 131.000.000.

### La convenzione c'è già nel tuo lavoro

In `2022 DDG 128 del 02-05-22` (impegno del riparto DM 464/2021) hai fatto esattamente
questo. Il CUP di Brescia `C81B21013200005` ha **due righe, una per PG**:

| CUP | Capitolo / PG | Importo impegnato (DD 128/2022) |
|---|---|---:|
| `C81B21013200005` | 7400 / 7 | € 333.059.208,74 |
| `C81B21013200005` | 7400 / 1 | € 26.486.678,33 |
| **somma** | | **€ 359.545.887,07** = importo **assegnato** dal DM 464/2021 |

E il totale dei due atti coincide al centesimo (€ 660.660.678,33): è la catena
assegnato → impegnato che vogliamo ricostruire. Anche in `2022 DD MIT 454-2022` hai
tenuto **due righe per due statuizioni di impegno** sullo stesso PG (€ 14.000.000 in
conto residui, art. 1; € 131.000.000 in competenza, art. 2) e **nessuna** quota annuale.
Il `141-2025` è l'unico documento in cui la riga segue lo stanziamento invece degli
impegni.

### Il principio: il registro è uno storico di eventi contabili *(precisato il 28/09)*

Registriamo gli **eventi contabili che prendono, tolgono o rimodulano risorse dello Stato
verso progetti**: un monitoraggio puntuale, con lo storico di tutte le variazioni. Ogni
atto che **dispone un'azione su un CUP** genera un evento nuovo, **anche se trascrive
quanto già fatto da un decreto precedente**. Due casi:

- un allegato che *"aggiorna e sostituisce"* quello di un decreto precedente (`342-2023`
  rispetto al DM 364/2021): ogni sua riga con CUP è un evento, anche se l'importo non è
  cambiato. Le righe del decreto sostituito restano nello storico, superate;
- due decreti che dispongono sullo stesso CUP e sullo stesso PG: sono due eventi, non un
  doppione.

Non è un evento ciò che l'atto **cita senza disporre**: somme che *"trovano copertura"* in
risorse già impegnate da altri atti, stanziamenti di legge, valore complessivo di una
convenzione, interventi solo dichiarati ammissibili. Nel `141-2025` l'art. 2 è tutto di
questo tipo. Gli eventi sono solo due: **l'impegno di due importi, da due PG diversi,
verso lo stesso CUP** (artt. 3 e 4). La difficoltà di estrazione sta qui: riconoscere,
fra cifre dello stesso rango, quelle che l'atto dispone e non si limita a richiamare.

### La regola

> Da un decreto di impegno si registra **una riga per ogni combinazione CUP × capitolo/PG
> impegnata da quell'atto**, con l'importo complessivo impegnato su quel PG. Si registra
> **ogni azione che l'atto dispone su un CUP**, anche se ne riproduce una precedente. Non
> si registrano: le quote annuali; il valore complessivo dell'intervento o della
> convenzione; le somme che l'atto cita come già impegnate da altri decreti senza
> disporre nulla su di esse; lo stanziamento di legge.

**Domande:**

**a)** Per il `141-2025` il tuo file diventa di due righe? — **deciso (Matteo, 28/09)**

- [X] Sì: `E51I04000010007` · 7419/2 · € 2.125.000.000 e `E51I04000010007` · 7416/1 · € 75.000.000

  Stesso CUP, due righe: la colonna capitolo/PG mostra che le risorse impegnate vengono
  da due PG diversi. Questo è il dato che ci interessa.

**b)** Stesso CUP e stesso PG, più statuizioni di impegno nello stesso atto (il caso del
`454-2022`: residui + competenza):

- [ ] Una riga per statuizione, come hai fatto (€ 14.000.000 e € 131.000.000)
- [ ] Una riga sola con la somma per PG (€ 145.000.000)

**c)** Un impegno unico per più CUP. Nel `128` il Comune di Torino ha
**€ 44.494.493,04 impegnati su 7400/7 per due CUP insieme** (`C11B21008720001`,
`D81E20000410007`). Nel tuo file li hai divisi a metà (€ 22.247.246,52 ciascuno), ma il
DM 464/2021 li aveva assegnati in misura diversa: **€ 9.408.502,67** e
**€ 35.085.990,37**. Per il registro degli impegni:

- [ ] Una riga sola con i due CUP e l'importo intero (impegno aggregato, non ripartito)
- [ ] Due righe con la ripartizione del decreto di assegnazione (9,4 M e 35,1 M)
- [ ] Due righe divise a metà, come ora — perché: ................................

**d)** Il tuo file ha ora due colonne in più, **CAPITOLO/PG** e **TIPOLOGIA** (assegnato,
impegnato, rimodulato, revocato, ...), compilate **solo per il `141-2025`**. È il livello
che vogliamo raggiungere: fonte dell'impegno (PG), importo e CUP, più la natura
dell'azione. Il punteggio le usa già dove sono compilate. Si possono compilare anche
sugli altri documenti, almeno sui decreti di impegno (`128`, `454`)?

- [ ] Sì, su tutto il file   - [ ] Sì, solo sui decreti di impegno   - [ ] No: ..........

> Perché pesa tanto: sul `141-2025` **tutti i modelli trovano già le due righe giuste**
> (7419/2 € 2.125.000.000 e 7416/1 € 75.000.000, marcate "impegnato"). Sbagliano perché
> gli abbiamo chiesto anche il totale per CUP e le quote annuali, e perché registrano le
> somme richiamate (contenuto non innovativo). La regola sopra toglie la concorrenza fra cifre dello stesso rango,
> che è la causa per cui i modelli piccoli crollano a 4-5 documenti su 16.

---

## Q4 — Su chi va imputato l'importo da 180 milioni?
*Foglio: `Q4 CUP 180 milioni`*

Sempre in `2022 DI Mims Mef 97 20.04.2022`, stesso importo di **€ 180.000.000**, due CUP
diversi:

| | CUP |
|---|---|
| Nel tuo file | `B31F20000030005` |
| Secondo il modello | `B44D20000040001` |

Il modello cita la riga "LINEA METROPOLITANA M1 PROLUNGAMENTO QUARTIERE BAGGIO – OLMI –
VALSESIA, Comune di Milano", colonna "FINANZIAMENTO AMMESSO (€)".

**Domanda:** quale dei due è corretto? (Se è il tuo, ci dici da quale riga l'hai preso?
Se ti accorgi che è una svista, va benissimo — ci serve solo saperlo.)

- [ ] `B31F20000030005`, riga: ................................
- [X] `B44D20000040001` — il modello ha ragione

---

## Q5 — Gli otto documenti senza righe
*Foglio: `Q5 senza riscontro`*

Otto PDF della cartella non compaiono nel tuo file. Abbiamo controllato cosa ci trova il
modello: su **sette non trova alcuna azione con CUP**, e sull'ottavo
(`2022 DI MIT-MEF 390-2022`) trova assegnazioni ma **nessun CUP**. Sembrano quindi atti che
davvero non contengono finanziamenti legati a progetti — `2022 DM 8.14-01-2022`, per dire,
è di due pagine e proroga solo un termine.

**Domanda:** confermi che non contengono azioni di finanziamento legate a CUP?

- [X] Sì, non c'era nulla da estrarre
- [ ] No, alcuni non li ho ancora lavorati: quali? ................................

> Ci serve per una ragione precisa: se confermi, quegli otto diventano un test di
> **precisione** — un modello che vi "trova" dei CUP sta inventando, ed è il tipo di errore
> che in un contesto di bilancio costa di più.

---

## Due segnalazioni minori
*Foglio: `Segnalazioni minori`*

1. **Una riga duplicata identica** (stesso documento, CUP e importo). Innocua per il
   confronto, la segnaliamo solo per completezza.
RISPOSTA ALLA SEGNALAZIONE 1: NON SI TRATTA DI UN ERRORE DI DUPLICAZIONE DI RIGA PERCHÈ QUELLA COPPIA CUP-IMPORTO COMPARE EFFETTIVAMENTE DUE VOLTE NEL DOCUMENTO.
2. In `2022 DM MIMS 342-2022` il codice **`PROV0000026838`** ha 14 caratteri e non è un CUP
   nel formato standard. Immaginiamo sia un codice provvisorio: confermi? Lo abbiamo già
   gestito nel confronto, ma vogliamo essere sicuri di aver capito bene.
RISPOSTA ALLA SEGNALAZIONE 2: CONFERMO CHE SI TRATTA DI UN CODICE PROVVISORIO.

---

## Decisione presa (Matteo, 28/09) — gli interventi solo ammissibili sono fuori perimetro

In `2022 DI Mims Mef 97 20.04.2022` l'**Allegato 3** elenca interventi *"valutati
ammissibili a finanziamento ma non rientranti nel piano di riparto"*, che *"saranno
finanziati seguendo lo scorrimento della graduatoria"* (art. 2). L'atto **non assegna
loro risorse**: non c'è un PG, non c'è un movimento. Nel registro del ponte PG → CUP non
entrano.

Le **9 righe** corrispondenti (€ 362.137.535, CUP `C11J22000000001`, `F47C19000350001`-`...390001`,
`F47D17000140005`, `B34J18000220001`, `B34J18000230001`) sono state tolte dal tuo file; la
versione precedente è in `0. Backup/`. Se ritrovi lo stesso caso in altri decreti, vale
la stessa regola.

---

## Cosa succede dopo

Con le tue risposte riscriviamo le istruzioni date ai modelli e rilanciamo la misura. Poi
il ciclo va avanti da solo — un modello terzo analizza le differenze residue e propone
correzioni, che però passano sempre da una nostra approvazione prima di essere applicate.

L'obiettivo finale non è "battere" il tuo lavoro: è capire **se un modello a pesi aperti,
installabile su macchine del Ministero, può arrivare al tuo livello**. Oggi il migliore
installabile su una singola scheda grafica sta a 9 documenti su 16, e il migliore su un
piccolo nodo a 11. Il tuo lavoro resta il metro di misura.
