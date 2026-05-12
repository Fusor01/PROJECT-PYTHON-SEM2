import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("mail_data.csv")
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})

X = df['Message']
y = df['Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction - Using n-grams for higher accuracy
tfidf = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, ngram_range=(1,2))
X_train_features = tfidf.fit_transform(X_train)
X_test_features = tfidf.transform(X_test)

# Using Random Forest for near-perfect accuracy
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_features, y_train)

# Evaluation
train_acc = accuracy_score(y_train, model.predict(X_train_features))
test_acc = accuracy_score(y_test, model.predict(X_test_features))

print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy: {test_acc*100:.2f}%")

# Save the model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
