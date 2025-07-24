# Import the necessary libraries
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    pipeline
)
import numpy as np
from evaluate import load as load_metric


### LOAD AND FINE-TUNE THE ROBERTa MODEL ###

# 1. Load CSVs (train.csv and validation.csv)
dataset = load_dataset(
    'csv',
    data_files={'train': 'startrain.csv', 'validation': 'starval.csv'},
    column_names=['text', 'label'],
    delimiter=','
)

# 2. Load in the dataset, model, and the task from Hugging Face
dataset = dataset.class_encode_column('label')
# Load the model and tokenizer
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Tokenize the dataset
def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
encoded = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=['text']
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=5,
    ignore_mismatched_sizes=True     
)

# 3. Setup training arguments
training_args = TrainingArguments(
    output_dir='./finetuned-stars',
    do_train=True,
    do_eval=True,
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
)

# 4. Load in all the epoch computation metrics from Hugging Face
accuracy_metric = load_metric('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# 5. Initialize the Trainer and train the model and evaluate using validation set
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded['train'],
    eval_dataset=encoded['validation'],
    compute_metrics=compute_metrics
)

# 6. Train the model
trainer.train()

# 7. Save the fine tuned model
trainer.save_model('roberta-stars-finetuned')
tokenizer.save_pretrained('roberta-stars-finetuned')



### RUN SAMPLE TESTS ###

# 1. Load in the fine-tuned model and tokenizer and set up the labels
config = AutoConfig.from_pretrained('roberta-stars-finetuned')
config.id2label   = {
    0: "1★", 1: "2★", 2: "3★", 3: "4★", 4: "5★"
}
config.label2id   = {v: k for k, v in config.id2label.items()}
model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-stars-finetuned', config=config
)
tokenizer = AutoTokenizer.from_pretrained('roberta-stars-finetuned')

# 2. Build a pipeline for star analyis
star_pipe = pipeline(
    "sentiment-analysis",   
    model=model,
    tokenizer=tokenizer
)

# 3. Write out tests 
examples = [
    "Absolutely phenomenal, I love it!",
    "It's okay, nothing special. But I like it.",
    "It's okay, nothing special.",
    "I don't know how I feel about this. Its bad but not terrible.",
    "Terrible movie",
]
for ex in examples:
    out = star_pipe(ex)[0]
    print(f"Example: {ex}")
    print(f"Predicted Star Val: {out['label']}, Confidence: {out['score']}")
    print()
