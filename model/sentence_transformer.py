import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class SentenceTransformer(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        outputs = self.backbone(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        return embeddings
