import torch
from model.multitask_model import MultiTaskModel

def run_inference():
  
    model = MultiTaskModel()
    model.eval()  # Set to eval mode

    # Example sentences
    sentences = ["This product is amazing!", "I hated the movie.", "The weather is okay."]

 
    with torch.no_grad():
        logits_a, logits_b = model(sentences)

        predictions_a = torch.argmax(logits_a, dim=1)
        predictions_b = torch.argmax(logits_b, dim=1)

    # Show predictions
    for i, sentence in enumerate(sentences):
        print(f"\nSentence: {sentence}")
        print(f"Task A (Classification) Prediction: Class {predictions_a[i].item()}")
        print(f"Task B (Sentiment) Prediction: {'Positive' if predictions_b[i].item() == 1 else 'Negative'}")
