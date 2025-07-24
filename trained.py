#Imports
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import numpy as np
from scipy.special import softmax
import csv
import urllib.request

import numpy as np
from evaluate import load as load_metric

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline


### 3-LABEL BASED SENTIMENT ANALYSIS (POS, NEG, NET) ###

# 1. Load CSVs (train.csv and validation.csv)
dataset = load_dataset(
    'csv',
    data_files={'train': 'train.csv', 'validation': 'val.csv'},
    column_names=['text', 'label'],
    delimiter=',',
)

# 2. Load the model and the task from Hugging Face
task = 'sentiment'
MODEL_ID = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 3. Set the labels for the dataset
labels_list = ['negative', 'neutral', 'positive']
dataset = dataset.class_encode_column('label')  

# 4. Tokenize the dataset
def tokenize_batch(batch):
    return tokenizer(
        batch["text"],        
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    
encoded_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=['text'],
)

# 5. Load the pretrained model (Twitter-RoBERTa)
num_labels = dataset['train'].features['label'].num_classes
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=3
)

# 6. Setup Fine-tuning arguments
training_args = TrainingArguments(
    output_dir='./finetuned-sentiment',
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


# 7. Load all metrics from Hugging Face
accuracy_metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# 8. Instantiate Trainer with the model and training arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# 9. Fine-tune the model
trainer.train()

# 11. Save the fine-tuned model and tokenizer
trainer.save_model('my-twitter-roberta-sentiment-finetuned')
tokenizer.save_pretrained('my-twitter-roberta-sentiment-finetuned')

# 12. Evaluate the model using EPOCH
metrics = trainer.evaluate()  
print(metrics)

# SAMPLE TESTING (POS, NEG, NET)
model_dir = "my-twitter-roberta-sentiment-finetuned"

# 1) create the labels for output
config = AutoConfig.from_pretrained(model_dir)
config.id2label   = {0: "negative", 1: "neutral", 2: "positive"}
config.label2id   = {"negative": 0, "neutral": 1, "positive": 2}

# 2. load the model and tokenizer
model     = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 3) build the sentiment analysis pipeline
sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 4) test it
examples = [
    "I love this movie!",
    "This is the worst film I've ever seen.",
    "It's okay, not great but not terrible either.",
    "I don't know how I feel about this.",
]

for example in examples:
    print(f"Example: {example}")
    print(sentiment(example))
    print()



