from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_dir = "my-twitter-roberta-sentiment-finetuned"

# 1) load and patch the config
config = AutoConfig.from_pretrained(model_dir)
config.id2label   = {0: "negative", 1: "neutral", 2: "positive"}
config.label2id   = {"negative": 0, "neutral": 1, "positive": 2}

# 2) reload model + tokenizer with patched config
model     = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 3) build the pipeline
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

