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

The LLM folder contains the Python files that allow training and testing of the LLM model, which detects the presence or absence of PHI in a text segment.

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
Contains the training, testing, and inference methods for the ClinicalT5 model. In particular, it also includes the method for plotting the ROC curve during testing.

### data_handler.py
Defines data handling during the training and testing phases, in particular dataset loading, cleaning, and splitting.

### pii_data_loader.py
Manages and tokenizes the datasets, from which it creates DataLoaders for training.

### training_clinicalT5-base.py
Trains the ClinicalT5 model on a dataset containing PHI and PII. For usage, consider the following options:

- Training strategy
   -b/--standard &emsp;train-validation-test split 
  
- -n/--num_splits &emsp;&nbsp;&nbsp;&nbsp; Number of  splits
- -e/--num_epochs &nbsp;&nbsp;&nbsp; Number of epoch
- --dataset &emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; Path of dataset.

Example

```bash
python3 training_clinicalT5-base.py -s -n 3 -e 5 --dataset datasets/training_dataset.csv --num_labels 2
```


### testing_clinicalT5-base.py
Performs testing of the model created during training. To use it:
```bash
python3 testing_clinicalT5-base.py
```


## ***NER***

### base_config.cfg
Configuration file that defines the hyperparameters, dataset paths, tokenization settings, and training options required to train the NER model.

### spacy-ner-spy.ipynb
Script used to train the NER model.

### test-ner.ipynb
Script used to test the previously trained NER model in order to evaluate its performance on a new test dataset.
