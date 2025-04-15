from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import re
from textblob import TextBlob
import os

app = Flask(__name__)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load model and feature columns if they exist
try:
    model = joblib.load('models/fake_email_model.pkl')
    feature_cols = joblib.load('models/feature_columns.pkl')
except FileNotFoundError:
    # If models don't exist, train them first
    from train_model import train_and_save_model
    train_and_save_model()
    model = joblib.load('models/fake_email_model.pkl')
    feature_cols = joblib.load('models/feature_columns.pkl')

def extract_features_from_input(email_data):
    features = {}
    
    # Content features
    content = email_data.get('content', '')
    features['content_length'] = len(content)
    features['word_count'] = len(content.split())
    features['link_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content))
    
    # Suspicious keywords
    suspicious_keywords = ['urgent', 'password', 'verify', 'account', 'suspended', 
                         'action required', 'immediately', 'click here', 'confirm', 'winner']
    content_lower = content.lower()
    for keyword in suspicious_keywords:
        features[f'contains_{keyword}'] = int(keyword in content_lower)
    
    # Sentiment analysis
    blob = TextBlob(content)
    features['sentiment_polarity'] = blob.sentiment.polarity
    features['sentiment_subjectivity'] = blob.sentiment.subjectivity
    
    # Subject features
    subject = email_data.get('subject', '')
    features['subject_uppercase_ratio'] = sum(1 for c in subject if c.isupper())/len(subject) if len(subject) > 0 else 0
    
    # Create DataFrame with all expected columns
    df = pd.DataFrame([features])
    
    # Ensure all expected columns are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_cols]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features
        features = extract_features_from_input(data)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0, 1]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)