from datetime import datetime

import torch
import argparse
import pandas as pd
import os
from custom_libs_clinical_t5_base.data_handler import DataHandler
from custom_libs_clinical_t5_base.pii_data_loader import PIIDataLoader
from custom_libs_clinical_t5_base.ClinicalT5_ModelTrainer import ClinicalT5_ModelTrainer


### Constants
METHODOLOGY = 1 # 1 for imbalanced training, 0 for balanced training
TORCH_RANDOM_SEED = 123
RANDOM_SEED = [ 1787671649, 1787671505] # Fixed seed for reproducible results
MODEL_NAME='hossboll/clinical-t5' # Model from Hugging Face
MODEL_RES='ClinicalT5-base_model'
OPTIMIZER_RES='optimizer'
SAVE_FOLDER='results_train' # Path where to save the data
WEIGHT_MODEL_PATH= r'results_train/ClinicalT5-base_model_tte.pt' # Model from Hugging Face

### Functions
def save_results_to_csv(results, filename):
    """Add results to a already created CSV file."""

    try:

        # Make new dir if dir does not exist
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        filepath = os.path.join(SAVE_FOLDER, filename)

        df = pd.DataFrame(results)

        # Add rows if file exist, else it will be created first
        if os.path.exists(filepath):
            df.to_csv(filepath, mode="a", header=False, index=False)
        else:
            df.to_csv(filepath, mode="w", header=True, index=False)

        print(f"Results saved to: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error while saving results: {e}")
        return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def training_test_validation(num_epochs, device, num_labels=2, random_seed=123):
    """ Performs standard train-test-validation split training on the dataset """


    # Set deterministic behavior for reproducible results
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(TORCH_RANDOM_SEED)

    # Initialize data handler and load the test dataset
    data_handler = DataHandler(DATASET_PATH, train_size=0.75, valid_size=0.0, test_size=0.25, seed=random_seed) # path of our datasets
    data_handler.load_data()
    data_handler.clean_data()

    # Split data into training, validation, and test sets
    if METHODOLOGY == 1:
        train_texts_gen, train_labels_gen, valid_texts_gen, valid_labels_gen, test_texts_gen, test_labels_gen = data_handler.split_data(dataset_type=True)
    else:
        train_texts_gen, train_labels_gen, valid_texts_gen, valid_labels_gen, test_texts_gen, test_labels_gen = data_handler.split_data(n_rows=1000, dataset_type=True)

    train_texts_temist, train_labels_temist, valid_texts_temist, valid_labels_temist, test_texts_temist, test_labels_temist = data_handler.split_data(n_rows=1000, dataset_type=False)

    train_texts = train_texts_gen + train_texts_temist
    train_labels = train_labels_gen + train_labels_temist

    valid_texts = (
            (valid_texts_gen if valid_texts_gen is not None else []) +
            (valid_texts_temist if valid_texts_temist is not None else [])
    )

    valid_labels = (
            (valid_labels_gen if valid_labels_gen is not None else []) +
            (valid_labels_temist if valid_labels_temist is not None else [])
    )

    test_texts = test_texts_gen + test_texts_temist
    test_labels = test_labels_gen + test_labels_temist

    # Create data loaders for batch processing
    data_loader = PIIDataLoader(train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels)
    train_loader, valid_loader, test_loader = data_loader.get_dataloader()

    # Initialize ClinicalT5 model trainer
    model_trainer = ClinicalT5_ModelTrainer(
        model_name=MODEL_NAME,
        save_folder=SAVE_FOLDER,
        device=device,
        num_labels=num_labels,
        num_epochs=num_epochs
    )

    # Train the model using training and validation sets
    model_trainer.train(train_loader, None)

    # Make sure the folder exists
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Save the model and optimizer state with the folder prefix
    methodology = 'Imbalanced' if METHODOLOGY==1 else "Balanced"

    model_trainer.save_model(
        os.path.join(SAVE_FOLDER, MODEL_RES + "_" + methodology + f'_tte.pt'),
        os.path.join(SAVE_FOLDER, OPTIMIZER_RES + "_" + methodology + f'_tte.pt')
    )

    # Generate and display training metrics plots
    # model_trainer.plot_metrics()

    data_loader_generated = PIIDataLoader(train_texts_gen, train_labels_gen, valid_texts_gen, valid_labels_gen, test_texts_gen, test_labels_gen)
    test_loader_generated = data_loader_generated.get_specific_dataloader("test")

    # Evaluate the trained model on the test set GENERATED
    test_results_generated = model_trainer.evaluate(test_loader_generated)

    # Prepare results for CSV

    results_generated = [{
        'method': 'standard_split',
        'methodology': 'Imbalanced' if METHODOLOGY==1 else "Balanced",
        'dataset':'GENERATED (Our)',
        'seed': random_seed,
        'fold': 1,
        'epochs': num_epochs,
        'test_accuracy': test_results_generated.get('accuracy', 0),
        'precision': test_results_generated.get('precision', 0),
        'recall': test_results_generated.get('recall', 0),
        'f1_score': test_results_generated.get('f1_score', 0),
        'auc_roc': test_results_generated.get('auc_roc', 0),
        'average_loss': test_results_generated.get('loss', 0),
        'training_time_min': test_results_generated.get('training_time_min', 0)
    }]

    # Save results to CSV
    save_results_to_csv(results_generated, 'validation_results.csv')


    data_loader_temist = PIIDataLoader(train_texts_temist, train_labels_temist, valid_texts_temist, valid_labels_temist, test_texts_temist, test_labels_temist)
    test_loader_temist = data_loader_temist.get_specific_dataloader("test")

    # Evaluate the trained model on the test set GENERATED
    test_results_temist = model_trainer.evaluate(test_loader_temist)

    # Prepare results for CSV

    results_temist = [{
        'method': 'standard_split',
        'methodology': 'Imbalanced' if METHODOLOGY==1 else "Balanced",
        'dataset':'ZENODO',
        'seed': random_seed,
        'fold': 1,
        'epochs': num_epochs,
        'test_accuracy': test_results_temist.get('accuracy', 0),
        'precision': test_results_temist.get('precision', 0),
        'recall': test_results_temist.get('recall', 0),
        'f1_score': test_results_temist.get('f1_score', 0),
        'auc_roc': test_results_temist.get('auc_roc', 0),
        'average_loss': test_results_temist.get('loss', 0),
        'training_time_min': test_results_temist.get('training_time_min', 0)
    }]

    # Save results to CSV
    save_results_to_csv(results_temist, 'validation_results.csv')

    return [results_generated, results_temist]

def test_zenodo_zero_exposure(device, num_labels=2):
    """ Performs test on all zenodo dataset """

    # Set deterministic behavior for reproducible results
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(TORCH_RANDOM_SEED)

    # Initialize data handler and load the test dataset
    data_handler = DataHandler(DATASET_PATH, train_size=0.75, valid_size=0.0, test_size=0.25) # path of our datasets
    data_handler.load_data()
    data_handler.clean_data()

    #Taking dataset zenodo
    _, _, _, _, test_texts, test_labels = data_handler.split_data(n_rows=1000, dataset_type=False)

    # Create data loaders for batch processing
    data_loader = PIIDataLoader(test_texts=test_texts, test_labels=test_labels)
    test_loader = data_loader.get_specific_dataloader("test")

    # Initialize ClinicalT5 model trainer
    model_trainer = ClinicalT5_ModelTrainer(
        model_name=MODEL_NAME,
        save_folder=SAVE_FOLDER,
        weight_model_path=WEIGHT_MODEL_PATH,
        device=device,
        num_labels=num_labels,
    )

    # Evaluate the trained model on the test set
    test_results = model_trainer.evaluate(test_loader)

    # Generate and display training metrics plots
    # model_trainer.plot_metrics()

    # Prepare results for CSV


    results = [{
        'method': 'standard_split',
        'dataset': 'ZENODO',
        'fold': 1,
        'test_accuracy': test_results.get('accuracy', 0),
        'precision': test_results.get('precision', 0),
        'recall': test_results.get('recall', 0),
        'f1_score': test_results.get('f1_score', 0),
        'auc_roc': test_results.get('auc_roc', 0),
        'average_loss': test_results.get('loss', 0),
    }]

    # Save results to CSV
    save_results_to_csv(results, 'zero-exposure_results_zenodo.csv')
    return results


def calculate_average_results(fold_results):
    """Calculate average metrics across all folds"""
    if not fold_results:
        return {}
    
    # Calculate averages for numeric fields
    numeric_fields = ['test_accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'average_loss', 'training_time_min']
    avg_result = {
        'method': fold_results[0]['method'] + '_average',
        'fold': 'average',
        'epochs': fold_results[0]['epochs']
    }
    
    for field in numeric_fields:
        values = [result[field] for result in fold_results if isinstance(result[field], (int, float))]
        avg_result[field] = sum(values) / len(values) if values else 0
    
    return avg_result

def arg_commandline():
    """ Create an argument parser object """
    parser = argparse.ArgumentParser(description="ClinicalT5 model training with different validation strategies")

    # Add parameters for dataloader
    parser.add_argument('-n', '--num_splits', type=int, default=5, help='number of splits')
    parser.add_argument('-e', '--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--dataset', type=str, default='Dataset/', help='Path to the dataset file')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of classification labels')  # Nuovo parametro

    return parser.parse_args()


### Main
def main():
    args = arg_commandline()

    global DATASET_PATH
    DATASET_PATH = args.dataset


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Number of labels: {args.num_labels}")

    print("Starting standard train-validation-test split training...")
    # results = training_test_validation(num_epochs=args.num_epochs, device=device, num_labels=args.num_labels, random_seed=RANDOM_SEED[METHODOLOGY])
    results_zenodo = test_zenodo_zero_exposure(device, args.num_labels)

if __name__ == "__main__":
    main()