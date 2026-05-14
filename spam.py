# Import libraries
import pandas as pd
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset 
data = {
    "message": [
        "Congratulations! You won a free lottery ticket",
        "Hey, are we meeting today?",
        "Claim your free cashback now",
        "Please send the project report",
        "Win money instantly by clicking here",
        "Let's go for lunch tomorrow",
        "Limited offer! Buy now",
        "Can you call me later?",
        "You have won a prize",
        "Meeting is scheduled at 5 PM"
    ],
    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])

    return text

# Apply cleaning
df["message"] = df["message"].apply(clean_text)

# Split Data
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ML Pipeline
# TF-IDF + Naive Bayes
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sample_message = ["Congratulations! You won a free iPhone"]

prediction = model.predict(sample_message)

print("\nCustom Message Prediction:", prediction[0])
joblib.dump(model, "spam_detector_model.pkl")

print("\nModel saved successfully as spam_detector_model.pkl")