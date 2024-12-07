import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
import joblib
from nltk.corpus import stopwords

nltk.download('stopwords')

# Laad en verwerk de data
def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            text, label = line.rsplit(maxsplit=1)  # Split de laatste woord (label)
            data.append(text.strip())
            labels.append(label.strip())
    return data, labels

# Train het model
def train_model(data, labels):
    # Split de data in training en testset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Maak een TF-IDF vectorizer om de tekst om te zetten naar numerieke waarden
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    # Transformeer de data naar TF-IDF representatie
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train het model (Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Maak voorspellingen en evalueer het model
    predictions = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")

    # Sla het model en vectorizer op
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

# Voorspelling functie
def predict_sentiment(text):
    # Laad het model en de vectorizer
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Zet de tekst om naar TF-IDF
    text_tfidf = vectorizer.transform([text])

    # Voorspel het sentiment
    prediction = model.predict(text_tfidf)

    return prediction[0]

# Hoofdfunctie
def main():
    # Voor het eerst trainen, gebruik je het bestand met gelabelde data
    # Verander het pad naar het bestand met trainingsdata
    file_path = 'sentiment_data.txt'

    # Laad de data en train het model
    print("Training the model...")
    data, labels = load_data(file_path)
    train_model(data, labels)

    # Vraag de gebruiker om een tekst in te voeren
    while True:
        input_text = input("\nEnter a text to check if it is negative or positive (or 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        result = predict_sentiment(input_text)

        # Geef het resultaat weer
        if result == "NEGATIVE":
            print("The sentiment is: NEGATIVE")
        else:
            print("The sentiment is: POSITIVE")

if __name__ == "__main__":
    main()
