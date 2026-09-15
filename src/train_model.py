from torch_geometric.loader import DataLoader
import os
import torch
import pandas as pd
from splits import DataSplitter
from GCN import GCN
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load saved graphs
graphs = torch.load(
    "../graphs/abide_pytorch_geometric_graphs.pt",
    weights_only=False
)

splitter = DataSplitter()

train_graphs, val_graphs, test_graphs = splitter.create_random_split(graphs)

print("Train:", len(train_graphs))
print("Validation:", len(val_graphs))
print("Test:", len(test_graphs))

# Create data loaders
train_loader = DataLoader(
    train_graphs,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_graphs,
    batch_size=4,
    shuffle=False
)

test_loader = DataLoader(
    test_graphs,
    batch_size=4,
    shuffle=False
)

# Create model
model = GCN()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

best_val_f1 = 0.0
patience = 15
epochs_without_improvement = 0
history = []
best_epoch = 0

# Training
for epoch in range(100):
    model.train()

    epoch_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        output = model(
            batch.x,
            batch.edge_index,
            batch.edge_weight,
            batch.batch
        )

        loss = criterion(output, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    average_loss = epoch_loss / n_batches

    # Validate the model
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in val_loader:

            output = model(
                batch.x,
                batch.edge_index,
                batch.edge_weight,
                batch.batch
            )

            predictions = output.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch.y.cpu().numpy()
            )

    # Calculate Metrics
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions
    )

    recall = recall_score(
        all_labels,
        all_predictions
    )

    f1 = f1_score(
        all_labels,
        all_predictions
    )

    # Save the best model based on f1 score
    if f1 > best_val_f1:

        best_val_f1 = f1
        best_epoch = epoch
        torch.save(
            model.state_dict(),
            "../models/best_gcn.pt"
        )
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch + 1}: "
        f"Loss={average_loss:.4f}, "
        f"Accuracy:  {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall:    {recall:.4f}, "
        f"F1 Score:  {f1:.4f},"
    )

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))

    history.append({
        "epoch": epoch,
        "train_loss": average_loss,
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1
    })

    # Early stopping check
    if epochs_without_improvement >= patience:
        print(
            f"Early stopping at epoch {epoch + 1}. "
            f"Best validation f1: {best_val_f1:.4f}"
        )

        break

# Save training results
history_df = pd.DataFrame(history)

history_df.to_csv(
    "../results/gcn_training_history_5.csv",
    index=False
)

# Load best model
model.load_state_dict(
    torch.load("../models/best_gcn.pt")
)

model.eval()

# Test
correct = 0
total = 0

with torch.no_grad():

    for batch in test_loader:

        output = model(
            batch.x,
            batch.edge_index,
            batch.edge_weight,
            batch.batch
        )

        predictions = output.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            batch.y.cpu().numpy()
        )

# Calculate Metrics
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

print()
print(
    f"Test: "
    f"Accuracy: {accuracy:.4f}, "
    f"Precision: {precision:.4f}, "
    f"Recall: {recall:.4f}, "
    f"F1 Score: {f1:.4f}"
)

print("Confusion matrix")
print(confusion_matrix(all_labels, all_predictions))

# Save test results
test_results = []
test_results.append({
        "best epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_confusion_matrix": confusion_matrix(all_labels, all_predictions)
    })

test_df = pd.DataFrame(test_results)

test_df.to_csv(
    "../results/gcn_test_results.csv",
    mode="a",
    header=not os.path.exists("../results/gcn_test_results.csv"),
    index=False
)