# 🎮 Game Review Analyzer

A web application that uses fine-tuned RoBERTa models to analyze game reviews for sentiment and star ratings.

## Features

- **Sentiment Analysis**: Predicts whether a review is positive, negative, or neutral
- **Star Rating**: Predicts a 1-5 star rating for game reviews
- **Modern Web Interface**: Beautiful, responsive UI with real-time analysis
- **Example Reviews**: Quick test buttons with sample reviews

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Models are Available**:
   - Make sure your fine-tuned models are in the correct directories:
     - `my-twitter-roberta-sentiment-finetuned/` (for sentiment analysis)
     - `roberta-stars-finetuned/` (for star ratings)

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Web Interface**:
   - Open your browser and go to: `http://localhost:8080`

## Usage

1. **Select Model Type**:
   - Click "Sentiment Analysis" for positive/negative/neutral classification
   - Click "Star Rating" for 1-5 star predictions

2. **Enter Review Text**:
   - Type or paste your game review in the text area
   - Use the example buttons to test with sample reviews

3. **Analyze**:
   - Click "Analyze Review" to get predictions
   - View the confidence score for each prediction

## API Endpoints

- `GET /`: Main web interface
- `POST /analyze`: Analyze text (JSON payload with `text` and `model_type`)
- `GET /health`: Check model availability status

## Model Information

- **Sentiment Model**: Fine-tuned RoBERTa for sentiment classification (negative/neutral/positive)
- **Star Rating Model**: Fine-tuned RoBERTa for 1-5 star rating prediction
- Both models are based on the `cardiffnlp/twitter-roberta-base-sentiment` architecture

## Example API Usage

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This game is amazing!", "model_type": "sentiment"}'
```

Response:
```json
{
  "text": "This game is amazing!",
  "prediction": "positive",
  "confidence": 0.9234,
  "model_type": "sentiment"
}
```

## Troubleshooting

- If models fail to load, check that the model directories exist and contain the necessary files
- Ensure you have sufficient RAM for loading the transformer models
- Check the `/health` endpoint to verify model status 