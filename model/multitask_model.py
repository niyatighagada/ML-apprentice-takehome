import torch.nn as nn
from model.sentence_transformer import SentenceTransformer

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_task_a=3, num_classes_task_b=2):
        super().__init__()
        self.encoder = SentenceTransformer()
        hidden_size = self.encoder.backbone.config.hidden_size

        self.task_a_head = nn.Linear(hidden_size, num_classes_task_a)
        self.task_b_head = nn.Linear(hidden_size, num_classes_task_b)

    def forward(self, sentences):
        embeddings = self.encoder(sentences)
        task_a_logits = self.task_a_head(embeddings)
        task_b_logits = self.task_b_head(embeddings)
        return task_a_logits, task_b_logits
