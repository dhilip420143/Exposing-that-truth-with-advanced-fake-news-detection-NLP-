import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load and balance dataset
df_fake = pd.read_csv("Fake.csv").sample(1000, random_state=42)
df_true = pd.read_csv("True.csv").sample(1000, random_state=42)

df_fake["label"] = 0  # Fake
df_true["label"] = 1  # Real

# Merge and shuffle
df = pd.concat([df_fake, df_true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
df = df.drop(["title", "subject", "date"], axis=1)

# Clean function (minimal for now)
def clean(text):
    return text.lower()

df["text"] = df["text"].astype(str).apply(clean)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
def predict_news(news):
    news = clean(news)
    vec = vectorizer.transform([news])
    pred = model.predict(vec)[0]
    return "✅ Real News" if pred == 1 else "❌ Fake News"

# Gradio App
gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=8, placeholder="Enter news text..."),
    outputs="text",
    title=" Fake News Detector (Balanced & Fixed)",
    description="Paste a news article to check if it's real or fake."
).launch()
