from torch_geometric.loader import DataLoader
import torch
from splits import DataSplitter
from GCN import GCN

# Load saved graphs
graphs = torch.load(
    "graphs/abide_pytorch_geometric_graphs.pt",
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

best_val_accuracy = 0.0

for epoch in range(20):
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

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in val_loader:

            output = model(
                batch.x,
                batch.edge_index,
                batch.edge_weight,
                batch.batch
            )

            predictions = output.argmax(dim=1)

            correct += (predictions == batch.y).sum().item()
            total += batch.y.size(0)

    val_accuracy = correct / total

    # Save the best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/best_gcn.pt"
        )

    print(
        f"Epoch {epoch + 1}: "
        f"Loss={average_loss:.4f}, "
        f"Val Accuracy={val_accuracy:.4f}"
    )

# Load best model
model.load_state_dict(
    torch.load("models/best_gcn.pt")
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

        correct += (predictions == batch.y).sum().item()
        total += batch.y.size(0)

test_accuracy = correct / total

print(f"Test Accuracy: {test_accuracy:.4f}")