import os
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def create_sample_data():
    data = {
        'content': [
            # Fake emails (label=1)
            "URGENT: Your account will be suspended! Click here: http://fake.com/verify",
            "Congratulations! You won $1000. Claim prize: http://scam.com/win",
            "Password reset required immediately: http://phish.com/reset",
            "Your bank account needs verification: http://steal.info/login",
            "Action required: Your subscription will renew for $499. Click to cancel: http://scam.net",
            "Security alert: Unusual login detected. Verify now: http://fake-login.com",
            "Your package delivery failed. Update details: http://tracking-scam.com",
            "Tax refund available! Claim now: http://irs-fake.gov",
            
            # Genuine emails (label=0)
            "Hi John, just checking in about our meeting tomorrow at 2pm.",
            "Hello team, attached is the quarterly report for your review.",
            "Your invoice #12345 is ready. Please find it attached.",
            "Meeting reminder: Project discussion at 3pm in Conference Room B",
            "Thanks for your application! We'll review it shortly.",
            "Your order #5678 has been shipped. Tracking number: XYZ123",
            "Monthly newsletter: Check out our latest updates!",
            "Password changed successfully for your account."
        ],
        'from': [
            # Fake senders
            "security@bank.com",
            "prizes@winners.org",
            "noreply@service.com",
            "support@yourbank.com",
            "billing@service.net",
            "security@account.com",
            "delivery@postalservice.com",
            "refunds@tax.gov",
            
            # Genuine senders
            "colleague@company.com",
            "manager@company.com",
            "billing@realcompany.com",
            "calendar@company.com",
            "hr@company.org",
            "orders@realstore.com",
            "newsletter@service.com",
            "accounts@service.com"
        ],
        'subject': [
            # Fake subjects
            "Account Suspension Notice",
            "You're a Winner!",
            "Password Reset Required",
            "Account Verification Needed",
            "Subscription Renewal Notice",
            "Security Alert",
            "Delivery Problem",
            "Tax Refund Available",
            
            # Genuine subjects
            "Meeting Reminder",
            "Quarterly Report",
            "Invoice #12345",
            "Project Discussion",
            "Application Received",
            "Order Shipped",
            "Monthly Newsletter",
            "Password Changed"
        ],
        'label': [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0]  # 1=fake, 0=genuine
    }
    return pd.DataFrame(data)

def extract_features(df):
    # Content features
    df['content_length'] = df['content'].apply(len)
    df['word_count'] = df['content'].apply(lambda x: len(x.split()))
    df['link_count'] = df['content'].apply(lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)))
    
    # Suspicious keywords
    suspicious_keywords = ['urgent', 'password', 'verify', 'account', 'suspended', 
                         'action required', 'immediately', 'click here', 'confirm', 'winner']
    for keyword in suspicious_keywords:
        df[f'contains_{keyword}'] = df['content'].str.lower().str.contains(keyword).astype(int)
    
    # Sentiment analysis
    df['sentiment_polarity'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_subjectivity'] = df['content'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Subject features
    df['subject_uppercase_ratio'] = df['subject'].apply(lambda x: sum(1 for c in x if c.isupper())/len(x) if len(x) > 0 else 0)
    
    return df

def train_and_save_model():
    # Create or load your dataset
    df = create_sample_data()
    
    # Feature engineering
    df = extract_features(df)
    
    # Prepare features and target
    feature_cols = ['content_length', 'word_count', 'link_count', 
                   'contains_urgent', 'contains_password', 'contains_verify',
                   'contains_account', 'contains_suspended', 'contains_click here',
                   'contains_winner', 'sentiment_polarity', 'sentiment_subjectivity',
                   'subject_uppercase_ratio']
    
    X = df[feature_cols]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    
    # Save model and feature columns
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fake_email_model.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    print("Model trained and saved successfully.")

if __name__ == '__main__':
    train_and_save_model()