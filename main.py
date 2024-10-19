from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding

print("Starting script...")

# Load the dataset from Hugging Face
print("Loading dataset...")
dataset = load_dataset("imanoop7/phishing_url_classification")
print(f"Dataset loaded. Keys: {dataset.keys()}")

# Check available splits and create train/validation/test splits if necessary
if "train" not in dataset or "validation" not in dataset or "test" not in dataset:
    print("Creating train/validation/test splits...")
    # Combine all available data
    combined_data = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]
    
    # Create train/validation/test splits
    splits = combined_data.train_test_split(test_size=0.2, seed=42)
    train_valid = splits["train"]
    test = splits["test"]
    splits = train_valid.train_test_split(test_size=0.1, seed=42)
    train = splits["train"]
    validation = splits["test"]
    
    dataset_dict = {
        "train": train,
        "validation": validation,
        "test": test
    }
else:
    print("Using existing dataset splits...")
    dataset_dict = dataset

print(f"Dataset splits: {dataset_dict.keys()}")

# Define pre-trained model path
model_path = "bert-base-uncased"

# Load model tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with binary classification head
print("Loading model...")
id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# Freeze base model parameters and unfreeze pooling layers
print("Freezing base model parameters...")
for name, param in model.base_model.named_parameters():
    param.requires_grad = False
    if "pooler" in name:
        param.requires_grad = True

# Define text preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Preprocess all datasets
print("Preprocessing datasets...")
tokenized_data = {}
for split, dataset in dataset_dict.items():
    print(f"Preprocessing {split} split...")
    tokenized_data[split] = dataset.map(preprocess_function, batched=True)
    print(f"{split} split preprocessed. Size: {len(tokenized_data[split])}")

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load metrics
print("Loading metrics...")
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
    positive_class_probs = probabilities[:, 1]
    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'], 3)
    predicted_classes = np.argmax(predictions, axis=1)
    acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'], 3)
    return {"Accuracy": acc, "AUC": auc}

# Define training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="bert-phishing-classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",  # This disables wandb logging
)

# Initialize trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()
print("Training completed.")

# Save the model and tokenizer
print("Saving model and tokenizer...")
model_save_path = "bert-phishing-classifier"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved to {model_save_path}")

# Evaluate on test dataset
print("Evaluating on test dataset...")
test_results = trainer.evaluate(tokenized_data["test"])
print("Test results:")
print(test_results)

print("Script completed successfully.")
