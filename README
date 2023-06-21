# Reinforcement Learning - Lunar Lander

Questo repository contiene l'implementazione di un algoritmo di apprendimento per rinforzo (Reinforcement Learning) utilizzando l'ambiente "LunarLander-v2" di Gym.

## Descrizione del Progetto

L'obiettivo di questo progetto è addestrare un agente di intelligenza artificiale a pilotare un modulo di atterraggio lunare nel gioco Lunar Lander. L'agente apprende attraverso l'interazione con l'ambiente, osservando lo stato attuale e prendendo azioni per massimizzare il punteggio ottenuto.

## Dipendenze

- Python 3.x
- PyTorch
- Gym

## Istruzioni

1. Installazione delle dipendenze:
`pip install torch gym`


2. Clonare il repository:
`git clone https://github.com/tuonome/reinforcement-learning-lunar-lander.git`


3. Navigare nella directory del progetto:
`cd reinforcement-learning-lunar-lander`


4. Eseguire il file di addestramento:
`python train.py`



## Struttura del Codice

Il codice è organizzato nei seguenti file:

- `utils.py`: Contiene funzioni di utilità per il calcolo della perdita, la selezione dell'azione e il tracciamento della curva di apprendimento.
- `network.py`: Contiene le definizioni delle reti neurali utilizzate per la politica e il valore dello stato.
- `train.py`: Il file principale che avvia l'addestramento dell'agente sul gioco Lunar Lander.

Durante l'addestramento, l'agente utilizza l'algoritmo Advantage Actor-Critic per migliorare le sue prestazioni nel gioco. La rete neurale per la politica (Policy Network) apprende a selezionare azioni basate sugli stati osservati, mentre la rete neurale per il valore dello stato (State Value Network) stima il valore di uno stato.

## Risultati

Durante l'addestramento, i punteggi medi ottenuti dall'agente vengono registrati e visualizzati periodicamente. Alla fine dell'addestramento, la curva di apprendimento viene tracciata e salvata come immagine `learning_curve.png`. Inoltre, i pesi delle reti neurali addestrate vengono salvati in file separati per la politica e il valore dello stato.

## Contributi

Sono benvenuti i contributi a questo progetto. Se desideri apportare modifiche o correzioni, ti invitiamo a creare una pull request.