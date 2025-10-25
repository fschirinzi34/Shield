# **Struttura codice**:
Qui brevemente viene spiegata la struttura del codice presente in questo repository organizzato nelle cartelle Dataset, LLM e NER

## ***Dataset***
In questa cartella è presente il codice relativo alla generazione del dataset di training, pulizia dello stesso e conversione da csv in formato json.

### No_PHI_Dataset_Generation.py
Script utilizzato per creare un dataset sintetico di dialoghi medico-paziente NON contenente PHI (Protected health information).

### PHI_Dataset_Generation.py
Script utilizzato per creare un dataset sintetico di dialoghi medico-paziente contenente PHI.

### create_json.ipynb
Script utilizzato per convertire il CSV creato in precedenza in un file JSON adatto all’addestramento del modello NER. Il file JSON presenta ogni esempio con il testo tokenizzato e un' etichettatura BIO corrispondente a ciascun token, così da rispettare il formato richiesto dal modello.

### uniq_clean_dataset.py
Script utilizzato per mitigare gli errori prodotti dal modello LLAMA:8B in fase di generazione del dataset

## ***LLM***


## ***NER***

### base_config.cfg
File di configurazione che definisce gli iperparametri, i percorsi dei dataset, le impostazioni di tokenizzazione e le opzioni di training necessarie per addestrare il modello NER

### spacy-ner-spy.ipynb
Script utilizzato per fare il training del modello NER

### test-ner.ipynb
Script utilizzato per testare il modello NER precedentemente addestrato, per verificarne le performance su un nuovo dataset di test
