from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Directory where the fine-tuned model is saved
MODEL_DIR = "roberta-stars-finetuned"

# 1) Load and patch the config for human-readable labels
config = AutoConfig.from_pretrained(MODEL_DIR)
# Ensure these match how you saved id2label in training
config.id2label = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
config.label2id = {v: k for k, v in config.id2label.items()}

# 2) Load the model and tokenizer with patched config
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 3) Build the pipeline
star_pipe = pipeline(
    "sentiment-analysis",  # classification task
    model=model,
    tokenizer=tokenizer
)

# 4) Function to test arbitrary texts

def predict_stars(text: str):
    """
    Predicts star rating for a given review text.
    Returns a tuple of (label, confidence).
    """
    result = star_pipe(text)[0]
    return result['label'], result['score']


# 5) Sample tests
if __name__ == "__main__":
# 14. Test examples
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