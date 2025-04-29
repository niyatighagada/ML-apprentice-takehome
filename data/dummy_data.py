import torch

def get_dummy_data():
    sentences = ["This is great!", "Terrible idea.", "I love this.", "Not good.", "Fantastic!", "Could be better."]
    labels_a = torch.tensor([0, 1, 0, 1, 0, 1])
    labels_b = torch.tensor([1, 0, 1, 0, 1, 0])
    return sentences, labels_a, labels_b
