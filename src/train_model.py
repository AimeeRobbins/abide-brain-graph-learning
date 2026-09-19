from torch_geometric.loader import DataLoader
import os
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from splits import DataSplitter
from GCN import GCN

class Trainer:
    """
    Trains, validates and tests a GCN on one split (random or one LOSO fold)
    """

    def split(self, held_out_site=None):
        # Load saved graphs
        graphs = torch.load("../graphs/abide_pytorch_geometric_graphs.pt", weights_only=False)

        splitter = DataSplitter()

        if held_out_site is None:
            train_g, val_g, test_g = splitter.create_random_split(graphs)
        else:
            train_g, val_g, test_g = splitter.create_loso_split(graphs, held_out_site)

        # Create data loaders
        train_loader = DataLoader(train_g, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_g, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_g, batch_size=4, shuffle=False)

        return train_loader, val_loader, test_loader

    def evaluate(self, model, loader):
        """Run the model on a loader and return (metrics dict, confusion matrix)."""
        model.eval()

        predictions = []
        labels = []
        probabilities = []

        with torch.no_grad():

            for batch in loader:
                output = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

                probabilities.extend(output.softmax(dim=1)[:, 1].cpu().numpy())
                predictions.extend(output.argmax(dim=1).cpu().numpy())
                labels.extend(batch.y.cpu().numpy())

        # Calculate Metrics dictionary
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
            "auc": roc_auc_score(labels, probabilities),
        }
        
        return metrics, confusion_matrix(labels, predictions)

    def train(self, model, train_loader, val_loader, tag):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_f1 = -1.0
        best_epoch = 0
        patience = 15
        epochs_without_improvement = 0
        history = []
        
        # Training
        for epoch in range(100):
            model.train()

            epoch_loss = 0.0

            for batch in train_loader:
                output = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                loss = criterion(output, batch.y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            average_loss = epoch_loss / len(train_loader)

            metrics, confusion_matrix = self.evaluate(model, val_loader)

            # Save the best model based on f1 score
            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                best_epoch = epoch

                torch.save(model.state_dict(), f"../models/best_gcn_{tag}.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch + 1}: "
                f"Loss={average_loss:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1 Score: {metrics['f1']:.4f}, "
                f"AUC: {metrics['auc']:.4f}"
            )

            print("Confusion Matrix:")
            print(confusion_matrix)

            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation f1: {best_val_f1:.4f}")
                break

        return best_epoch + 1, best_val_f1

    def test(self, model, test_loader, best_epoch, best_val_f1, tag):
        # Load best model
        model.load_state_dict(torch.load("../models/best_gcn.pt"))
        model.eval()

        correct = 0
        total = 0

        metrics, confusion_matrix = self.evaluate(model, test_loader)

        print(
                f"Tag: {tag} "
                f"Best Epoch {best_epoch}: "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1 Score: {metrics['f1']:.4f}, "
                f"AUC: {metrics['auc']:.4f}"
        )

        print("Confusion Matrix:")
        print(confusion_matrix)

        tn, fp, fn, tp = confusion_matrix.ravel()

        return {
                "tag": tag,
                "n_test": int(confusion_matrix.sum()),
                "best epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "test_accuracy": metrics['accuracy'],
                "test_balanced_accuracy": metrics['balanced_accuracy'],
                "test_precision": metrics['precision'],
                "test_recall": metrics['recall'],
                "test_f1": metrics['f1'],
                "test_auc": metrics['auc'],
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }   

    def run(self, held_out_site=None):
        tag = held_out_site or "random"

        # Create model
        model = GCN()
        train_loader, val_loader, test_loader = self.split(held_out_site)
        best_epoch, best_val_f1 = self.train(model, train_loader, val_loader, tag)
        return self.test(model, test_loader, best_epoch, best_val_f1, tag)    