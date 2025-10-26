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

Nella cartella LLM sono contenuti i file python che permettono di addestrare e testare il modello LLM
che riconosce la presenza o l'assenza di PHI in una porzione di testo

```bash
LLM/
├── custom_libs_clinical_t5_base/
│   ├── ClinicalT5_ModelTrainer.py
│   ├── data_handler.py
│   └── pii_data_loader.py
├── testing_clinicalT5-base.py
└── training_clinicalT5-base.py
```

### ClinicalT5_ModelTrainer.py
Contiene i metodi di training, di testing e di inferenza per il modello ClinicalT5. In particolare è presente anche il metodo per il plotting della curva ROC in fase di test. 

### data_handler.py
Definisce la gestione dei dati nelle fasi di addestramento e di testing, in particolare caricamento, pulizia e split dei dataset

### pii_data_loader.py
Gestisce e tokenizza i dataset, da cui crea i DataLoader per il training.

### training_clinicalT5-base.py
Addestra il modello ClinicalT5 su un dataset contenente PHI e PII. Per l'utilizzo considerare le seguenti opzioni:
- Strategia di training
  - -b/--standard &emsp;train-validation-test split 
  - -k/--kfold &emsp;&emsp;&nbsp;&nbsp;&nbsp;K-fold cross-validation
  - -s/--strkfold &emsp;&nbsp;&nbsp;Stratified K-fold cross-validation
- -n/--num_splits &emsp;&nbsp;&nbsp;&nbsp; Numero di splits
- -e/--num_epochs &nbsp;&nbsp;&nbsp; Numero di epoche
- --dataset &emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; Path del dataset.

Esempio

```bash
python3 training_clinicalT5-base.py -s -n 3 -e 5 --dataset datasets/training_dataset.csv --num_labels 2
```


### testing_clinicalT5-base.py
Effettua il test del modello creato dall'addestramento. Per utilizzarlo:
```bash
python3 testing_clinicalT5-base.py
```


## ***NER***

### base_config.cfg
File di configurazione che definisce gli iperparametri, i percorsi dei dataset, le impostazioni di tokenizzazione e le opzioni di training necessarie per addestrare il modello NER

### spacy-ner-spy.ipynb
Script utilizzato per fare il training del modello NER

### test-ner.ipynb
Script utilizzato per testare il modello NER precedentemente addestrato, per verificarne le performance su un nuovo dataset di test
