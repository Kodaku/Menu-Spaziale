# Domanda
Raccontaci di un progetto Generative AI in cui hai lavorato.
Concentrati su:
a. Gli attori coinvolti e le persone con cui hai collaborato
b. Le feature che hai implementato
c. La roadmap che hai seguito

# Risposta

Il progetto riguardava un POC di un chatbot implementato all'intero di una soluzione di creazione di sessioni di firma elettronica. L'obiettivo era esplorare gli AI Agents a basso livello (LangChain e LangGraph).

## b. Feature implementate

Il chatbot si occupava di rispondere alle seguenti domande:
1. Fornire informazioni su un firmatario.
2. Fornire informazioni su un template di firma.
3. Fornire informazioni sulle sessioni firmate da un firmatario in un periodo di tempo.
4. Creare sessioni di firma a partire da un template.
5. Filtrare tutti i discorsi che esulavano da 1-4.

Il chatbot non rispondeva mai con testo in linguaggio naturale ma con informazioni strutturate su quanto richiesto.

### Architettura

Di base la soluzione seguiva i seguenti step:
1. Ricezione della domanda dell'utente.
2. Questa veniva letta da un nodo supervisor (su cui stava un LLM) che decideva il task.
3. In teoria, deciso il task, avrebbe dovuto anche decidere in autonomia i nodi da chiamare per completare il task.
4. Ciascuna foglia rappresentava un microtask che raccoglieva informazioni necessarie al completamento di un task più grande. La query dell'utente veniva tradotta in un filtro per interrogare un db già esistente.
5. Il sistema a quel punto si sarebbe reso conto quando il task era completato e avrebbe resituito una risposta strutturata da visualizzare nella schermata del chatbot.

## c. Roadmap

A questo punto la roadmap seguita è stata la seguente:
1. Studio e formazione per apprenere le tecnologie dietro l'utilizzo degli AI Agents.
2. Presentazione al team dei frutti della formazione. Con il team leader è stato svolto un bel brainstorming per capire come poter strutturare il POC.
3. Realizzazione del POC secondo l'architettura supervisor-executor (non so se c'è nome ufficiale, noi l'abbiamo chiamata così) interamente basata su LLM.
4. Evidenza che gli LLM locali disponibili erano inaffidabili.
5. Step intermedio con utilizzo di LLM tramite API per rendere possibile la realizzazione del chatbot (frontend e backend).
6. Training di un modello di NER su un task semplice (es. info sul firmatario X).
7. Confronto LLM vs NER.
8. Rimpiazzo dell'LLM con il modello di NER per quel task specifico.
9. Presentazione interna all'azienda.

Potendo contare solo su LLM locali, piccoli, non velocissimi (per risorse limitate sulla mia macchina) e che allucinavano spesso, la soluzione si è spostata molto velocemente dall'utilizzo di un LLM per parsare la richiesta dell'utente all'utilizzo di un modello di Name Entity Recognition. Non avendo chiaro inoltre il contesto d'uso il dataset di training del modello di NER è stato costruito sulla base di frasi suggerite da vari LLM (GPT, Claude, Gemini, Deepseek, ecc.). L'unica casistica in cui l'LLM funzionava bene era sulla scelta della tipologia di task richiesto. Ma anche qui, avendo a disposizione i dataset per la NER si è pensato al training di un classificatore.

Quindi in sintesi l'architettura c'era ma gli LLM no, o meglio ci sono stati in un primo momento. Sono in stati rimpiazzati alla fine con modelli più leggeri e affidabili per quel caso d'uso.

## a. Attori coinvolti

Per la soluzione mi sono occupato della sua realizzazione full stack (il servizio con l'Agent era un back-back end, poi c'era il backend vero e proprio in .NET e il frontend in Angular). Ho partecipato alla sua relizzazione, quando necessario, con gli altri componenti del team se c'era da sistemare qualcosa su backend o frontend e con il team leader per avere di feedback sulla direzione della soluzione.

Il POC è stato presentato internamente all'azienda al CEO e altri vertici, che hanno, alla luce dei limiti emersi e delle soluzioni proposte per farne fronte, optato per la non adozione del chatbot. Il know how è stato però apprezzato e ha permesso ad altri reparti di poter implementare soluzioni simili con più risorse a disposizione.
