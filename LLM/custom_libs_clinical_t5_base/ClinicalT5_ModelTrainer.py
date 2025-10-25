import os
import time
import torch
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5ForSequenceClassification
from transformers import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

class ClinicalT5_ModelTrainer:
    """ Trainer per ClinicalT5-base usando T5ForSequenceClassification """

    def __init__(self, model_name, save_folder, device, weight_model_path=None, num_labels=2, num_epochs=3, lr=5e-5, loss_fn=None):
        self.model_name = model_name
        self.weight_model_path = weight_model_path
        self.device = device
        self.num_epochs = num_epochs
        self.save_folder = save_folder
        self.lr = lr
        self.num_labels = num_labels
        
        # Inizializza il tokenizer per ClinicalT5-base
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Inizializza il modello ClinicalT5 per sequence classification
        # Usa T5ForSequenceClassification con il numero di label specificato
        # Aggiunto from_flax=True per caricare dai pesi Flax
        self.model = T5ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        if weight_model_path is not None:
            state_dict = torch.load(weight_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            # Inizializza l'optimizer
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)


        self.model.to(device)
        
        # Inizializza le liste per tracciare le metriche
        self.train_losses = []
        self.valid_accuracies = []
        
        # Imposta la funzione di loss di default se non fornita
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        
        # Crea directory per salvare i plot
        self.plots_dir = save_folder
        self._create_plots_directory()

    def _create_plots_directory(self):
        """Crea directory per salvare i plot se non esiste."""
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            print(f"Created directory: {self.plots_dir}")

    def _get_timestamp(self):
        """Ottieni timestamp corrente per nominare i file."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def train(self, train_loader, valid_loader=None):
        """ Addestra il modello ClinicalT5 sui dati di training forniti """

        start_time = time.time()
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Imposta il modello in modalità training
            self.model.train()
            batch_losses = []
            start_time_for = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                
                # Sposta i dati del batch sul device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass - ora usa direttamente T5ForSequenceClassification
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Backward pass e ottimizzazione
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_losses.append(loss.item())

                # Stampa progresso ogni 50 batch
                if not batch_idx % 50:
                    end_time = time.time()
                    iter_time = end_time - start_time_for
                    print(f'Epoch: {epoch+1:04d}/{self.num_epochs:04d} | '
                        f'Batch {batch_idx:04d}/{len(train_loader):04d} | '
                        f'Loss: {loss:.4f} | '
                        f'Time_batch: {iter_time:.4f} s | ')

            # Calcola la loss media per l'epoca
            epoch_loss = sum(batch_losses) / len(batch_losses)
            self.train_losses.append(epoch_loss)

            # Calcola l'accuracy di validazione se il loader di validazione è fornito
            if valid_loader is not None:
                valid_accuracy = self.compute_accuracy(valid_loader)
                self.valid_accuracies.append(valid_accuracy)
                print(f'Epoch: {epoch+1:04d}/{self.num_epochs:04d} | '
                    f'Training Loss: {epoch_loss:.4f} | '
                    f'Validation Accuracy: {valid_accuracy:.2f}%')
            else:
                # Per scenari di k-fold cross-validation
                print(f'Epoch: {epoch+1:04d}/{self.num_epochs:04d} | '
                    f'Training Loss: {epoch_loss:.4f} | '
                    f'No validation step in this epoch.')

            print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')

        print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')

    def compute_accuracy(self, data_loader):
        """ Calcola l'accuracy sul dataset fornito """

        self.model.eval()
        correct_pred, num_examples = 0, 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Sposta i dati del batch sul device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted_labels = torch.max(outputs.logits, 1)

                # Aggiorna i contatori
                num_examples += labels.size(0)
                correct_pred += (predicted_labels == labels).sum()
        
        accuracy = correct_pred.float() / num_examples * 100
        return accuracy.item()

    def plot_roc_curve(self, data_loader, save_plot=True, filename=None):
        """ Plotta e opzionalmente salva la curva ROC per classificazione binaria """

        print("Generating ROC curve...")
        self.model.eval()
        y_true = []
        y_scores = []

        with torch.no_grad(): 
            for batch in data_loader: 
                # Sposta i dati del batch sul device
                input_ids = batch['input_ids'].to(self.device) 
                attention_mask = batch['attention_mask'].to(self.device) 
                labels = batch['labels'].to(self.device) 

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Raccogli le label vere e le probabilità predette
                y_true.extend(labels.cpu().numpy()) 
                y_scores.extend(outputs.logits.softmax(dim=1)[:, 1].cpu().numpy())

        # Calcola i componenti della curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores) 
        roc_auc = auc(fpr, tpr)

        # Crea il plot della curva ROC
        plt.figure(figsize=(10, 7)) 
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.5f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.fill_between(fpr, tpr, color='darkorange', alpha=0.3)

        # Personalizza il plot
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Salva il plot se richiesto
        if save_plot:
            if filename is None:
                filename = f"roc_curve_{self._get_timestamp()}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {filepath}")

        plt.show()

    def save_model(self, model_path, optimizer_path):
        """ Salva i state dictionary del modello e dell'optimizer """

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optim.state_dict(), optimizer_path)
        print(f"Model saved to: {model_path}")
        print(f"Optimizer saved to: {optimizer_path}")

    def evaluate(self, test_loader, save_plot=True, filename=None):
        """ Valutazione completa del modello sui dati di test """

        print("Starting model evaluation...")
        self.model.eval()
        all_preds = []
        all_labels = []
        all_scores = []
        total_loss = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(test_loader):
                # Sposta i dati del batch sul device
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                logits = outputs.logits

                # Calcola la loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                # Ottieni predizioni e punteggi di probabilità
                _, preds = torch.max(logits, 1)
                scores = torch.softmax(logits, dim=1)[:, 1] # Probabilità per classe positiva

                # Raccogli tutte le predizioni e label
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

        # Calcola le metriche di valutazione
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Calcola AUC-ROC solo se sono presenti più classi
        if len(set(all_labels)) > 1:
            auc_roc = roc_auc_score(all_labels, all_scores)

            # Calcola i componenti della curva ROC
            fpr, tpr, _ = roc_curve(all_labels, all_scores)

            # Crea e visualizza la curva ROC
            plt.figure(figsize=(10, 7))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.5f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.fill_between(fpr, tpr, color='purple', alpha=0.2)
            
            # Personalizza il plot
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic (ROC) Curve - Test Set')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            # Salva il plot se richiesto
            if save_plot:
                if filename is None:
                    filename = f"test_roc_curve_{self._get_timestamp()}.png"
                filepath = os.path.join(self.plots_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Test ROC curve saved to: {filepath}")
            
            plt.show()
        else:
            auc_roc = None

        avg_loss = total_loss / len(test_loader)

        # Stampa i risultati della valutazione
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Test Accuracy: {accuracy * 100:.5f}%")
        print(f"Precision: {precision:.5f}")
        print(f"Recall: {recall:.5f}")
        print(f"F1-Score: {f1:.5f}")
        if auc_roc is not None:
            print(f"AUC-ROC: {auc_roc:.5f}")
        else:
            print("AUC-ROC: Not calculable (less than 2 classes)")
        print(f"Average Loss: {avg_loss:.4f}")
        print("="*50)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "loss": avg_loss
        }

    def load_model(self, model_path, optimizer_path):
        """ Carica i state dictionary del modello e dell'optimizer """

        # Carica lo state dict del modello
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f"Model loaded from: {model_path}")

        # Inizializza l'optimizer con i parametri correnti
        self.optim = AdamW(self.model.parameters(), lr=self.lr, correct_bias=True)
        
        try:
            # Carica lo state dict dell'optimizer
            optimizer_state_dict = torch.load(optimizer_path, map_location=self.device)

            # Controlla se lo state dict ha le chiavi richieste
            if 'state' in optimizer_state_dict and 'param_groups' in optimizer_state_dict:
                # Gestisci la chiave 'correct_bias' mancante (problema di compatibilità)
                if 'correct_bias' not in optimizer_state_dict['param_groups'][0]:
                    print("Warning: 'correct_bias' key missing in optimizer state dict.")
                
                self.optim.load_state_dict(optimizer_state_dict)
                print(f"Optimizer loaded successfully from: {optimizer_path}")
            else:
                raise KeyError('Optimizer state dict is missing required keys.')
                
        except KeyError as e:
            print(f"Key error while loading optimizer state: {e}")
            print("Reinitializing optimizer with default parameters.")
            self.optim = AdamW(self.model.parameters(), lr=self.lr, correct_bias=True)

    def plot_metrics(self, save_plot=True, filename=None):
        """ Plotta le metriche di training (loss e accuracy di validazione) """
        if not self.train_losses:
            print("No training data to plot. Train the model first.")
            return
            
        plt.figure(figsize=(15, 6))
        
        # Plotta la training loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 
                'b-', marker='o', linewidth=2, markersize=6, label='Training Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Plotta l'accuracy di validazione se disponibile
        if self.valid_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(self.valid_accuracies) + 1), self.valid_accuracies, 
                    'g-', marker='s', linewidth=2, markersize=6, label='Validation Accuracy')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
        else:
            # Se non ci sono dati di validazione, mostra un messaggio
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, 'No Validation Data Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title('Validation Accuracy', fontsize=14, fontweight='bold')

        plt.tight_layout()
        
        # Salva il plot se richiesto
        if save_plot:
            if filename is None:
                filename = f"training_metrics_{self._get_timestamp()}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Training metrics plot saved to: {filepath}")
        
        plt.show()

    def get_training_summary(self):
        """ Ottieni un riassunto del processo di training """
        if not self.train_losses:
            return {"message": "No training data available"}
            
        summary = {
            "total_epochs": len(self.train_losses),
            "final_training_loss": self.train_losses[-1],
            "best_training_loss": min(self.train_losses),
            "worst_training_loss": max(self.train_losses),
        }
        
        if self.valid_accuracies:
            summary.update({
                "final_validation_accuracy": self.valid_accuracies[-1],
                "best_validation_accuracy": max(self.valid_accuracies),
                "worst_validation_accuracy": min(self.valid_accuracies)
            })
            
        return summary

# Esempio di utilizzo diretto come nel tuo esempio
def test_clinical_t5_classification():
    """Test di esempio per verificare che il modello funzioni correttamente"""
    
    # Inizializza tokenizer e modello
    model_name = "hossboll/clinical-t5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Test con un esempio di testo medico
    text = "Patient presents with chest pain and shortness of breath"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    print(f"Predicted class: {predicted_class_id}")
    print(f"Logits: {logits}")
    
    # Test con label per calcolare la loss
    labels = torch.tensor([1])  # Esempio di label
    loss = model(**inputs, labels=labels).loss
    print(f"Loss: {round(loss.item(), 4)}")
    
    return model, tokenizer

if __name__ == "__main__":
    # Test del modello
    print("Testing ClinicalT5-base for sequence classification...")
    test_clinical_t5_classification()
