from model.sentence_transformer import SentenceTransformer
from train.train_loop import train_model
from inference import run_inference

if __name__ == "__main__":
    # Task 1: Show sentence embeddings
    print("----- Task 1: Sentence Transformer Embedding Output -----")
    model = SentenceTransformer()
    sentences = ["The sky is blue.", "I love machine learning.", "This is a test sentence."]
    embeddings = model(sentences)
    print("Sample Embeddings Shape:", embeddings.shape)

    # Task 4: Simulate training
    print("\n----- Task 4: Training Loop Output -----")
    train_model()

    # Run Inference
    print("\n----- Inference Output -----")
    run_inference()
