# 🚀 Unified Fitness Calculator (UFC)

## Obiettivo del Calcolatore
L'obiettivo primario della **Unified Fitness Calculator (UFC)** è valutare l'efficacia e l'efficienza di un Prompt di Sistema evoluto, traducendo la performance in un unico valore numerico da **minimizzare (Loss)**.

## 🧠 Metrica Fondamentale: Valutazione Semantica
A differenza delle metriche lessicali (come ROUGE o Keyword matching), UFC si basa esclusivamente sulla **Similarità Coseno degli Embeddings**.

### Dettagli Tecnici:
* **Modello Embedding**: `sentence-transformers/all-mpnet-base-v2`.
* **Vantaggio**: Il sistema premia il **significato corretto** anche se la formulazione della risposta (o del reasoning) è sintatticamente diversa dalla soluzione di riferimento.

## ⚖️ Componenti della Loss Totale

La Loss totale è data dalla somma di quattro componenti pesate:

$$
\text{Loss}_{\text{Totale}} = \text{Costo}_{\text{Accuracy}} + \text{Costo}_{\text{Rationale}} + \text{Costo}_{\text{Token}} + \text{Costo}_{\text{Tempo}}
$$

| Componente | Peso di Default | Ruolo nell'Evoluzione |
| :--- | :--- | :--- |
| **$\text{Costo}_{\text{Accuracy}}$** | $w_{\text{acc}} = 10.0$ | Penalità massima in caso di **risposta semanticamente errata** (Massima priorità). |
| **$\text{Costo}_{\text{Rationale}}$** | $w_{\text{rat}} = 5.0$ | Penalità per **ragionamento logicamente errato**. |
| **$\text{Costo}_{\text{Token}}$** | $w_{\text{tok}} = 0.01$ | Costo per la **verbosità**. Spinge verso risposte concise. |
| **$\text{Costo}_{\text{Tempo}}$** | $w_{\text{time}} = 0.1$ | Costo per la **latenza**. |

---

## 🔑 Osservazioni Critiche e Tuning Finale

Il processo di testing ha portato a un'ottimizzazione cruciale del calcolatore per garantire la **stabilità evolutiva**.

### 1. La Soglia Semantica Differenziata (Achieved)

Il punto di forza del calcolatore è l'implementazione di soglie di tolleranza dinamiche:

* **Risposta (Ans)**: $\text{Threshold} = \text{0.8}$ (Alto rigore).
* **Ragionamento (Rat)**: $\text{Threshold} = \text{0.6}$ (Maggiore tolleranza).

**Impatto**: L'abbassamento della soglia del Ragionamento a $\text{0.6}$ (ottenuto con $0.8 - 0.2$) ha risolto il problema dei "falsi negativi" nei testi complessi (Caso 6), garantendo che la Loss sia azzerata per i concetti validi, anche se espressi con grande variazione lessicale.

### 2. Successo nell'Isolamento dell'Errore (Caso 5)

Il calcolatore è in grado di **isolare l'errore logico** anche quando la risposta finale è corretta (Risposta OK, Rationale FAIL). Questo è fondamentale per evolvere prompt