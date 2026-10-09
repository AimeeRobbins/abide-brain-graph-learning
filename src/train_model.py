from torch_geometric.loader import DataLoader
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
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
from GCN1 import GCN1
from sampler import MultiSiteBatchSampler

def coral_loss(source, target):
    d = source.size(1)
    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    cov_source = (source_c.T @ source_c) / (source.size(0) - 1)
    cov_target = (target_c.T @ target_c) / (target.size(0) - 1)
    return (cov_source - cov_target).pow(2).sum() / (4 * d * d)

class Trainer:
    """
    Trains, validates and tests a GCN on one split (random or one LOSO fold)
    """

    def split(self, held_out_site=None, use_coral_sampler=False, coral_batch_size=16, coral_sites_per_batch=2):
        # Load saved graphs
        graphs = torch.load("../graphs/abide_pytorch_geometric_graphs.pt", weights_only=False)

        splitter = DataSplitter()

        if held_out_site is None:
            train_g, val_g, test_g = splitter.create_random_split(graphs)
        else:
            train_g, val_g, test_g = splitter.create_loso_split(in_graphs=graphs, held_out_site=held_out_site)

        # Create data loaders, dependent on CORAL
        if use_coral_sampler:
            sampler = MultiSiteBatchSampler(train_g, batch_size=coral_batch_size, sites_per_batch=coral_sites_per_batch)
            train_loader = DataLoader(train_g, batch_sampler=sampler)
        else:
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001) # could add weight decay: weight_decay=0.01

        best_val_auc = -1.0
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

            history.append({
                "epoch": epoch + 1,
                "train_loss": average_loss,
                "val_auc": metrics["auc"],
                "val_f1": metrics["f1"],
            })

            min_epochs_before_saving = 5

            # Save the best model based on auc score
            if epoch >= min_epochs_before_saving and metrics["auc"] > best_val_auc:
                best_val_auc = metrics["auc"]
                best_epoch = epoch

                torch.save(model.state_dict(), f"../models/best_gcn_{tag}.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation auc: {best_val_auc:.4f}")
                break

        return best_epoch + 1, best_val_auc, history

    def train_coral(self, model, train_loader, val_loader, tag, coral_weight=0.5):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

        best_val_auc = -1.0
        best_epoch = 0
        patience = 15
        epochs_without_improvement = 0
        min_epochs_before_saving = 5
        history = []

        for epoch in range(100):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                embeddings = model.get_graph_embedding(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                logits = model.classifier(embeddings)
                cls_loss = criterion(logits, batch.y)

                sites_in_batch = batch.site
                unique_sites = list(set(sites_in_batch))

                coral_total = 0.0
                pairs = 0
                for i in range(len(unique_sites)):
                    for j in range(i + 1, len(unique_sites)):
                        mask_i = [s == unique_sites[i] for s in sites_in_batch]
                        mask_j = [s == unique_sites[j] for s in sites_in_batch]
                        emb_i = embeddings[mask_i]
                        emb_j = embeddings[mask_j]
                        if emb_i.size(0) > 1 and emb_j.size(0) > 1:
                            coral_total += coral_loss(emb_i, emb_j)
                            pairs += 1

                coral_term = coral_total / max(pairs, 1)
                loss = cls_loss + coral_weight * coral_term

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            average_loss = epoch_loss / len(train_loader)
            metrics, confusion_matrix = self.evaluate(model, val_loader)

            history.append({
                "epoch": epoch + 1,
                "train_loss": average_loss,
                "val_auc": metrics["auc"],
                "val_f1": metrics["f1"],
            })

            if epoch >= min_epochs_before_saving and metrics["auc"] > best_val_auc:
                best_val_auc = metrics["auc"]
                best_epoch = epoch
                torch.save(model.state_dict(), f"../models/best_gcn_{tag}.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation auc: {best_val_auc:.4f}")
                break

        return best_epoch + 1, best_val_auc, history

    def test(self, model, test_loader, best_epoch, best_val_auc, tag):
        # Load best model
        model.load_state_dict(torch.load(f"../models/best_gcn_{tag}.pt"))
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
                "best_val_auc": best_val_auc,
                "test_accuracy": metrics['accuracy'],
                "test_balanced_accuracy": metrics['balanced_accuracy'],
                "test_precision": metrics['precision'],
                "test_recall": metrics['recall'],
                "test_f1": metrics['f1'],
                "test_auc": metrics['auc'],
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }   

    def plot_history(self, history, tag):
        epochs = [h["epoch"] for h in history]
        losses = [h["train_loss"] for h in history]
        aucs = [h["val_auc"] for h in history]

        fig, ax1 = plt.subplots()
        ax1.plot(epochs, losses, color="tab:red", label="train loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss", color="tab:red")

        ax2 = ax1.twinx()
        ax2.plot(epochs, aucs, color="tab:blue", label="val AUC")
        ax2.set_ylabel("val AUC", color="tab:blue")

        plt.title(f"Training curve: {tag}")
        plt.savefig(f"../results/loss_curve_{tag}.png")
        plt.close()

    def run(self, held_out_site=None, model_class=GCN, model_tag="gcn"):
        tag = f"{held_out_site or 'random'}_{model_tag}"

        # Create model
        model = model_class()
        train_loader, val_loader, test_loader = self.split(held_out_site)
        best_epoch, best_val_auc, history = self.train(model, train_loader, val_loader, tag)

        self.plot_history(history, tag)

        return self.test(model, test_loader, best_epoch, best_val_auc, tag)    

    def run_coral(self, held_out_site=None, model_class=GCN, model_tag="coral_gcn", coral_weight=0.5):
        tag = f"{held_out_site or 'random'}_{model_tag}"
        model = model_class()
        train_loader, val_loader, test_loader = self.split(held_out_site, use_coral_sampler=True)
        best_epoch, best_val_auc, history = self.train_coral(model, train_loader, val_loader, tag, coral_weight=coral_weight)

        self.plot_history(history, tag)

        return self.test(model, test_loader, best_epoch, best_val_auc, tag)

    def run_repeated_loso(self, sites, model_class, model_tag, n_repeats=5, use_coral=False, coral_weight=0.5):
        all_results = []

        for seed in range(n_repeats):
            torch.manual_seed(seed)
            np.random.seed(seed)

            for site in sites:
                if use_coral:
                    result = self.run_coral(held_out_site=site, model_class=model_class, model_tag=f"{model_tag}_seed{seed}", coral_weight=coral_weight)
                else:
                    result = self.run(held_out_site=site, model_class=model_class, model_tag=f"{model_tag}_seed{seed}")
                result["site"] = site
                result["seed"] = seed
                result["model"] = model_tag
                all_results.append(result)

        return all_results