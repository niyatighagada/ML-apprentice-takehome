# ML Apprentice Take-Home Project



Objective

The goal of this project is to implement, train, and optimize neural network architectures focusing on transformers and multi-task learning (MTL). Below is my solution with detailed explanations and justifications for each design choice.

Task 1: Sentence Transformer Implementation

Approach

I implemented a sentence transformer using the distilbert-base-uncased model from HuggingFace Transformers. This choice was made because DistilBERT offers a good trade-off between performance and efficiency.

Key Design Choices

Backbone: Used a pre-trained transformer (distilbert-base-uncased) to leverage existing language understanding.

Embedding Extraction: Selected the CLS token output (outputs.last_hidden_state[:, 0]) to represent the entire sentence as a fixed-length embedding.

Framework: Chose PyTorch due to its flexibility and ease of debugging.

Testing

I validated the model by encoding a few sample sentences and confirming that embeddings of expected shape (batch_size x hidden_size) were obtained.

Task 2: Multi-Task Learning Expansion

Approach

I expanded the Sentence Transformer into a Multi-Task Learning (MTL) model by adding two separate task-specific heads:

Task A: Sentence Classification (3 arbitrary classes)

Task B: Sentiment Analysis (binary classification)

Key Architecture Changes

Retained the same shared encoder for both tasks to promote feature sharing.

Added two separate fully-connected layers for classification:

task_a_head: Maps embeddings to 3 output classes.

task_b_head: Maps embeddings to 2 output classes.

Outputs logits for both tasks during a single forward pass.

Reasoning

MTL allows the model to learn a shared representation beneficial across tasks, improving generalization and reducing overfitting.

Task 3: Training Considerations

Scenario Analysis

1. Freezing the Entire Network:

Only new heads train.

Useful when the dataset is very small to avoid overfitting.

Limitation: No adaptation to new domain data.

2. Freezing Only the Transformer Backbone:

The heads can specialize in tasks without modifying general language understanding.

Ideal when pre-trained backbone is already sufficient for sentence understanding.

3. Freezing Only One Head:

Useful when one task is stable and another needs further learning.

Common in continual learning or multi-phase training.

Transfer Learning Strategy

Pretrained Model Chosen: distilbert-base-uncased

Light, fast, and effective for general NLP tasks.

Layers to Freeze/Unfreeze:

Initially freeze the backbone during early epochs.

Later unfreeze selectively to allow fine-tuning if needed.

Rationale:

Start by leveraging pre-trained knowledge.

Gradually allow learning new domain-specific nuances.

Reduces risk of catastrophic forgetting.

Task 4: Training Loop Implementation (BONUS)

Assumptions and Decisions

Used dummy data simulating input sentences and class labels.

Computed separate CrossEntropyLoss for each task.

Aggregated losses as simple addition (could also weight differently if needed).

Used a shared optimizer (Adam) across all parameters.

Focus Areas

Hypothetical Data:

Sentences and labels are fabricated for demonstration purposes.

Forward Pass:

Single forward pass through encoder.

Branch into two heads for dual outputs.

Metrics:

For production, metrics like Accuracy, F1-score, Precision/Recall would be added separately for each task

How to Run the Project

# Install dependencies
pip install -r requirements.txt

# Run the training script
python main.py

Requirements

See requirements.txt for dependencies




