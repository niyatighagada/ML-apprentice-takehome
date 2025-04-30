import torch
from torch import nn, optim
from model.multitask_model import MultiTaskModel
from data.dummy_data import get_dummy_data
from sklearn.metrics import accuracy_score, f1_score

def train_model():
    model = MultiTaskModel()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    sentences, labels_a, labels_b = get_dummy_data()

    for epoch in range(3):
        optimizer.zero_grad()

        # Forward pass
        logits_a, logits_b = model(sentences)

        # Loss
        loss_a = criterion(logits_a, labels_a)
        loss_b = criterion(logits_b, labels_b)
        loss = loss_a + loss_b
        loss.backward()
        optimizer.step()

        # Predictions
        preds_a = torch.argmax(logits_a, dim=1)
        preds_b = torch.argmax(logits_b, dim=1)

        # Metrics
        acc_a = accuracy_score(labels_a.numpy(), preds_a.numpy())
        f1_a = f1_score(labels_a.numpy(), preds_a.numpy(), average='weighted')

        acc_b = accuracy_score(labels_b.numpy(), preds_b.numpy())
        f1_b = f1_score(labels_b.numpy(), preds_b.numpy(), average='weighted')

        # Log output
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, "
              f"Task A - Acc: {acc_a:.2f}, F1: {f1_a:.2f}, "
              f"Task B - Acc: {acc_b:.2f}, F1: {f1_b:.2f}")
