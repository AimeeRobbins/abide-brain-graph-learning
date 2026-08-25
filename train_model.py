train_loader = DataLoader(
    graphs,
    batch_size=4,
    shuffle=True
)

model = GCN()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

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

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / n_batches:.4f}")
