from flask import Flask, render_template, request, jsonify
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os

app = Flask(__name__)

# Load sentiment model
def load_sentiment_model():
    # Try fine-tuned model first
    model_dirs = [
        "finetuned-sentiment/checkpoint-189",
        "finetuned-sentiment/checkpoint-3",
        "my-twitter-roberta-sentiment-finetuned"
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
            try:
                config = AutoConfig.from_pretrained(model_dir)
                config.id2label = {0: "negative", 1: "mixed", 2: "positive"}
                config.label2id = {"negative": 0, "mixed": 1, "positive": 2}
                
                model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                
                return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            except Exception as e:
                print(f"Failed to load model from {model_dir}: {e}")
                continue
    
    # Fallback to base model
    try:
        print("Using base sentiment model as fallback")
        # Load the base model with proper label mapping
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        
        # Set up proper label mapping
        model.config.id2label = {0: "negative", 1: "mixed", 2: "positive"}
        model.config.label2id = {"negative": 0, "mixed": 1, "positive": 2}
        
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Failed to load base sentiment model: {e}")
        return None

# Load star rating model
def load_star_model():
    # Try fine-tuned model first
    model_dirs = [
        "finetuned-stars/checkpoint-189",
        "roberta-stars-finetuned"
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
            try:
                config = AutoConfig.from_pretrained(model_dir)
                config.id2label = {0: "1★", 1: "2★", 2: "3★", 3: "4★", 4: "5★"}
                config.label2id = {v: k for k, v in config.id2label.items()}
                
                model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                
                return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            except Exception as e:
                print(f"Failed to load model from {model_dir}: {e}")
                continue
    
    # Fallback to base model with custom labels
    try:
        print("Using base model with custom star labels as fallback")
        # Create a simple mapping function for star ratings
        def star_predictor(text):
            # Use sentiment to approximate star rating
            sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
            sentiment_result = sentiment_pipe(text)[0]
            sentiment = sentiment_result['label']
            confidence = sentiment_result['score']
            
            # Map sentiment to stars
            if sentiment == 'positive':
                stars = "4★" if confidence > 0.8 else "3★"
            elif sentiment == 'negative':
                stars = "1★" if confidence > 0.8 else "2★"
            else:
                stars = "3★"
            
            return {'label': stars, 'score': confidence}
        
        return star_predictor
    except Exception as e:
        print(f"Failed to load base star model: {e}")
        return None

# Initialize models
print("Loading sentiment model...")
sentiment_pipeline = load_sentiment_model()
print("Loading star model...")
star_pipeline = load_star_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()
    model_type = data.get('model_type', 'sentiment')
    
    if not text:
        return jsonify({'error': 'Please provide some text to analyze'})
    
    try:
        if model_type == 'sentiment' and sentiment_pipeline:
            if callable(sentiment_pipeline) and not hasattr(sentiment_pipeline, '__call__'):
                # Handle custom star predictor
                result = sentiment_pipeline(text)
            else:
                result = sentiment_pipeline(text)[0]
            return jsonify({
                'text': text,
                'prediction': result['label'],
                'confidence': round(result['score'], 4),
                'model_type': 'sentiment'
            })
        elif model_type == 'stars' and star_pipeline:
            # Handle custom star predictor
            result = star_pipeline(text)
            return jsonify({
                'text': text,
                'prediction': result['label'],
                'confidence': round(result['score'], 4),
                'model_type': 'stars'
            })
        else:
            return jsonify({'error': f'{model_type} model not available'})
    except Exception as e:
        return jsonify({'error': f'Error analyzing text: {str(e)}'})

@app.route('/health')
def health():
    return jsonify({
        'sentiment_model_loaded': sentiment_pipeline is not None,
        'star_model_loaded': star_pipeline is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 