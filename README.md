> **Disclaimer**: La soluzione è stata realizzata con l'ausilio di Claude Code. Una volta completato tutto il codice mi sono occupato di revisionarlo per capire cosa fosse stato fatto, commentare i punti clue e individuare qualche stortura nel codice (soprattutto errori logici), visto che comunque tutto funzionava senza errori.
> Non lavorando continuamente a soluzioni GenAI (alla fine il POC realizzato risale ad 1 anno fa e non comprendeva il RAG) ma più su soluzioni di Computer Vision e report di valutazione di modelli di AI ho preferito puntare su una soluzione il più semplice possibile per me da comprendere e di cui proporre miglioramenti sensati.

## Soluzioni proposte

La seguente repository propone due soluzioni al test tecnico:

1. **Soluzione con DB strutturato (senza RAG)**: in questo caso si costruisce, con l'ausilio di un LLM un db statico composto dai file json che si possono trovare nella cartella src/kb. Il modo in cui è fatto si basa sulla struttura delle domande presentate. La domanda viene letta e mappata in un filtro da un LLM (Llama3.1-8b o Llama3.3-70b utilizzati tramite l'API di Groq). Tale filtro è utilizzato per interrogare il db. Le submission di questa soluzione si trovano nei file *submission_llama3.1-8b* e *submission_llama3.3-70b*. Sebbene sia la migliore delle due risulta essere poco flessibile in caso di aggiunta di nuove domande insieme ai limiti di utilizzo degli LLM tramite API. Lo score Jaccard complessivo ottenuto è circa del 65%

2. **Soluzione RAG con BM25**: in questo caso la parte di retrieval non è gestita con un db vettoriale ma con l'algoritmo BM25 che ricerca in un corpus i documenti che meglio si integrano con la query. La parte di augmentation è gestita inserendo i k documenti estratti nel prompt dell'LLM con la query. Infine la parte di generation è l'LLM che genera la risposta come una serie di numeri con gli id dei piatti. Siccome è difficile gestire le casistiche di filtri negativi (es. piatti che non contengono X) in quanto si rischia di estrarre alla peggio tutto il corpus di testo che andrebbe a rendere non indifferente la grandezza del prompt dell'LLM, la soluzione proposta opta per una riduzione dei documenti coerenti con la domanda. Riguardo le parole negative del filtro si è inoltre deciso di rimuoverle solo per la parte di retrieve per evitare che BM25 trovi sì i documenti che menzionano X ma poi possa utilizzare la parola di negazione come mezzo per estrarre altri documenti non contestuali alla domanda. Rimuovendo la negazione si è sicuri di includere i piatti che menzionano X e poi di includere anche altri piatti rilevanti. Lo score Jaccard complessivo è intorno al 20%.

**Nota**: L'affermazione sulle negazioni è stata suggerita da Claude. Non ho evidenze che una soluzione con db vettoriale con embedding non riesca  a cogliere la semantica delle negazioni. L'unica cosa su cui ho un po' di certezze è che se si vogliono trovare piatti senza un ingrediente X e quest'ultimo è presente solo in 1 piatto su 300 allora si dovrebbe avere K=almeno 299 e quindi un prompt gigantesco sull'LLM conivolto nella parte di generation. La soluzione con db invece avrebbe un prompt molto leggero (solo per mappare la domanda in un filtro) e poi tutto il resto è deterministico.

## Processo decisionale

La soluzione con RAG partendo da embedding è stata comunque considerata ma, visto il poco tempo a disposizione si è preferito procedere in questo modo:
1. Brainstorming delle possibili soluzioni (con Claude).
2. Analisi dei problemi delle soluzioni.
3. Ranking delle soluzioni.
4. Implementazione della migliore, sulla carta (con db). Il task si presentava come un problema di filtering, data la natura delle domande.
5. Implementazione di una soluzione RAG semplice (con BM25).

## Soluzioni alternative considerate

Sono state considerate anche queste soluzioni:
1. Soluzione RAG con db vettoriale (quindi embedding di chunk di testo): la soluzione con db ha proposto un modo intelligente (anche se forse un po' fragile) di fare chunking dei documenti. La perplessità sui filtri negativi rimane.
2. Soluzione con agenti per le domande più complesse: presenza di un supervisor che decide gli agenti da chiamare. Ciascun agente è esperto di un dominio: codice galattico, distanze pianeti, blog, manuale di cucina, menu vari (vista la varietà dei menù non è escluso di avere un altro supervisor sui menu che fa lavorare N agenti).

## Possibilità di espansione della soluzione

1. Combinare la soluzione con db a quella con BM25 (o con db vettoriale) in una sorta di Ensemble. Richiederebbe un sistema di quality check per evitare di accorpare in modo cieco le risposte.
2. Introdurre un sistema di "cross-validation" per individuare gli iperparametri ottimali (soprattuto nel caso d'uso di un RAG).
3. Introdurre un sistema di monitoraggio delle domande per capirne la semantica, il dominio trattato e fanre data exploration per capire meglio come sono fatte. In questo modo si può avere maggiormente chiaro quanto l'introduzione di nuove domande può spostare l'accuratezza della soluzione.

   Avere chiara la struttura semantica delle domanda può, in un mondo ideale, far svincolare la soluzione dall'utilizzo degli LLM per rispondere alle domande. La soluzione con db mostra infatti buoni risultati con un metodo quasi deterministico. Spostare il parsing della query da farlo con un LLM a farlo con un NER può abbattere i costi di utilizzo sebbene richieda di investire molto tempo nella progettazione e realizzazione. Gli LLM sarebbero utilizzati solo per costruire la kb ed eventualmente per costruire un dataset di training per il modello di Name Entity Recognition.

4. Introdurre un sistema di monitoraggio sull'utilizzo degli LLM: consumo di token totale, consumo medio di token per domanda, ecc.
5. Rendere il sistema generalizzabile all'utilizzo di diversi LLM. Nella soluzione attuale una grande debolezza è che, appena si raggiungono i limiti sull'API di Groq bisogna aspettare che si resettino. Questo rallenta molto il testing.
6. Rendere il sistema, in caso sia messo a disposizione verso l'esterno, robusto verso attacchi di prompt injection.

# Utilizzo
Si consiglia di lasciare tutto così com'è in quanto la soluzione si basa sull'API di Groq con tutti i limiti di utilizzo.
In caso si disponga di un API key seguire questi step:
1. Creare un file .env a livello di root e nominare la variabile GROQ_API_KEY. Qui inserire l'API key.
2. Nel file constants.py dare alla variabile API_KEY_ENV_NAME il valore del nome della varibile d'ambiente appena creata al punto 1.
3. Se si volessero ricreare tutti i file .md dei pdf eseguire il notebook convert_pdf_to_md. Si consiglia di non farlo in quanto alcuni file sono stati modificati a mano per far sì che non riempissero il contesto di llama3.1-8b-instant.
4. Se si volesse ricreare il db intero lanciare nell'ordine i file, tutti nella cartella "no_rag": parse_manuale.py, parse_codice.py, parse_blogs.py, extract_menus.py. In questi file si può modificare il codice per indicare di utilizzare un modello Llama piuttosto che un altro.
5. Per creare il file delle submission no_rag lanciare lo script pipeline.py. Anche qui si può modificare il codice per modificare l'LLM.
6. Per quanto riguarda la soluzione rag: lanciare prima build_index.py e poi pipeline.py
7. Per validare lanciare lo script evaluation.py o evaluation_by_difficulty.py in questo modo: "pyhon evaluation.py --submission ../submission_no_rag_llama3.3-70b.csv". La sintassi per il file evaluation_by_difficulty.py è la stessa.
