import torch
from torch import nn, optim
from model.multitask_model import MultiTaskModel
from data.dummy_data import get_dummy_data

def train_model():
    model = MultiTaskModel()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    sentences, labels_a, labels_b = get_dummy_data()

    for epoch in range(3):
        optimizer.zero_grad()
        logits_a, logits_b = model(sentences)
        loss_a = criterion(logits_a, labels_a)
        loss_b = criterion(logits_b, labels_b)
        loss = loss_a + loss_b
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
