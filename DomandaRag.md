# Domanda
Scenario: devi sviluppare una soluzione RAG.
Quali ragionamenti faresti nella scelta delle tecnologie da utilizzare, modelli e moduli, tenendo conto dei requisiti di latenza, costi, privacy, scalabilità e qualità delle risposte?
Supponi di avere a disposizione la knowledge base e un set di domande di esempio a cui la RAG dovrebbe essere in grado di rispondere con alta accuratezza.

# Risposta

## Analisi del dominio e privacy

Innanzitutto cercherei di avere chiaro il dominio del problema, se c'è.
Questo mi aiuterebbe a capire in prima battuta se ci possono essere di mezzo dati sensibili, come dati aziendali, di dipendenti, ecc.

## Embedding e Retrieval

Supponiamo che il RAG sia fatto tramite ricerca di embedding.
A questo punto ramificherei il modo di estrarre gli embedding:
1. Se non ci sono dati sensibili si può anche pensare a una soluzione che fa uso di API di terze parti, come OpenAI.
2. Se ci sono dati sensibili allora o l'azienda garantisce che è possibile utilizzare API di terze parti in quanto è presente un account aziendale (es. Azure) e quindi garantiscono loro che i dati che saranno utilizzati non saranno in nessun modo utilizzati dal servizio di terze parti; oppure è necessario adottare un modello di embedding locale o comunque interno all'azienda (es. sui server aziendali).

Una volta stabilito come estrarre gli embedding si deve scegliere come effettuare la parte di Retrieval, quindi come salvare gli embedding:
1. Utilizzare un db vettoriale.
2. Salvarli su un file, esempio .pickle o .npy.

La scelta dipende in base alla quantità di dati presente.

**Nota**: la parte di retrieval si può fare anche con una baseline più semplice come BM25 (zero costi di chiamate a modelli di embedding sia economici che di latenza) e poi confrontare la soluzione a embedding con quella senza.

## Augmented Generation e Scalabilità

La parte di AG (Augmented Generation) si può fare con un LLM, dipende sempre dalla tipologia di dati coinvolti (sensibili o non sensibili) e vale lo stesso discorso del calcolo degli embedding.
Questa parte presenta anche un problema non indifferente di scalabilità per i seguenti motivi:
0. Quanti utenti sono previsti? Si è stimato un RPM medio (o al 95esimo percentile)?
1. Se l'LLM lo si usa con API bisogna tenere presente dei limiti di utilizzo, consumi di token, disponibilità del modello a X richieste al minuto, ecc. Lo studio dell'LLM più adatto al caso è maggiormente incentrato sulla parte di availability e costi piuttosto che sulla parte di accuratezza delle risposte.
Inoltre la scelta dell'LLM stesso può portarsi costi non indifferenti.
2. Se l'LLM è locale bisogna innanzitutto sincerarsi di avere le risorse a disposizione, di base una GPU. Supponendo di avere a disposizione le risorse allora ci si può concentrare più sulla parte di availability e accuratezza (spesso i modelli locali sono leggeri e molto proni ad allucinare). Più le risorse a disposizione sono maggiori e più si può spostare l'attenzione agli stessi punti del punto 1 (i costi sarebbero di setup e manutenzione dell'architettura).
3. Fatta la scelta si consiglia uno stress test per andare a stimare al meglio quanto il sistema reggerebbe, quando inizia a degenerare, quali costi comporterebbe un carico di un certo tipo, ecc. Le modalità di stress test dipendono dalla soluzione adottata.

**In sintesi, per esperienza:**
- **LLM su API** → alta qualità delle risposte ma richiede attenzione a costi, limiti di utilizzo e che fine faranno i dati.
- **LLM locale** → possibile bassa qualità delle risposte, sempre disponibile (a meno di crash dei server su cui gira), i dati restano in azienda.

## Implementazione

Una volta progettata la soluzione da questi punti di vista si può iniziare l'implementazione:
1. **Soluzione con API**: posso utilizzare un numero minimale di moduli, es. Openai, Groq, ecc. La soluzione non richiede alta competenza.
2. **Soluzione locale**: tanti moduli necessari, come LangChain, Huggingfacehub, pytorch (o tensorflow) se ad esempio sono necessari dei ritocchi a dei modelli open source, LangGraph (se prevedo la presenza di Agenti), MLFlow (se devo fare dei fine-tuning o devo tracciare delle valutazioni senza avere grafici sparsi per cartelle). Richiede competenze più avanzate oltre che tempi di sviluppo non indifferenti.
