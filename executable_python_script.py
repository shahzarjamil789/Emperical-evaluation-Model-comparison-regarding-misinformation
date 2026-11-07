"""
Fake News Model Comparison - Automated Empirical Evaluation Pipeline
Author: Shahzar Jameel (Department of Computer Science, Alhamd Islamic University)
Repository: https://github.com/shahzarjameel/fake-news-model-comparison
License: MIT
"""

import re
import time
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Ensure NLTK resources are safely downloaded
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception:
    # Fallback default stopwords list in case of network/SSL issues
    stop_words = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"}

def generate_synthetic_benchmark_data(num_samples=1000):
    """
    Generates a deterministic synthetic news dataset mimicking the Kaggle dataset schema
    if local CSV file is not present.
    """
    print("[INFO] Generating baseline dataset for verification execution...")
    np.random.seed(42)
    authors = ["John Doe", "Jane Smith", "Anonymous", "Reuters Tech", "State Media", "FactChecker Org"]
    titles_real = [
        "Economic Growth Reaches New Peak According to Central Bank Report",
        "New Climate Change Policy Approved by International Summit",
        "Scientific Breakthrough in Renewable Solar Energy Technology",
        "Global Infrastructure Initiative Announced at Trade Conference"
    ]
    titles_fake = [
        "SHOCKING: Secret Alien Technology Discovered in Abandoned Vault!",
        "Miracle Cure Sweeping the Nation hidden by Pharmaceutical Giants",
        "Banned Video Proves Global Conspiracy Behind Financial Crisis",
        "You Won't Believe What Celebrity Did Last Night - Click Here!"
    ]
    
    data = []
    for i in range(num_samples):
        label = np.random.choice([0, 1])
        author = np.random.choice(authors)
        if label == 0:
            title = np.random.choice(titles_real)
            text = "Official reports confirm legitimate events unfolding with verifiable data."
        else:
            title = np.random.choice(titles_fake)
            text = "Unverified rumors and wild claims circulating online without evidence."
        data.append({"id": i, "title": title, "author": author, "text": text, "label": label})
    return pd.DataFrame(data)

def preprocess_text(df):
    """
    Metadata fusion, regex cleaning, lowercasing, stopword removal, and Porter Stemming.
    """
    print("[INFO] Executing metadata fusion and morphological preprocessing pipeline...")
    df['author'] = df['author'].fillna('')
    df['title'] = df['title'].fillna('')
    df['content'] = df['author'] + ' ' + df['title']
    
    ps = PorterStemmer()
    
    cleaned_contents = []
    for content in df['content']:
        cleaned = re.sub('[^a-zA-Z]', ' ', content)
        cleaned = cleaned.lower()
        tokens = cleaned.split()
        stemmed = [ps.stem(word) for word in tokens if word not in stop_words]
        cleaned_contents.append(' '.join(stemmed))
        
    df['cleaned_content'] = cleaned_contents
    return df

def run_evaluation_pipeline(X, y, test_size=0.33, random_state=42):
    """
    Trains and evaluates Multinomial Naive Bayes, Logistic Regression, and Passive-Aggressive Classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 1. TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 2. Bag-of-Words Vectorization
    bow = CountVectorizer(max_features=5000)
    X_train_bow = bow.fit_transform(X_train)
    X_test_bow = bow.transform(X_test)
    
    models = {
        "Passive-Aggressive (PAC) [TF-IDF]": (PassiveAggressiveClassifier(max_iter=50, random_state=random_state), X_train_tfidf, X_test_tfidf),
        "Logistic Regression (LR) [TF-IDF]": (LogisticRegression(max_iter=200, random_state=random_state), X_train_tfidf, X_test_tfidf),
        "Multinomial Naive Bayes (MNB) [TF-IDF]": (MultinomialNB(), X_train_tfidf, X_test_tfidf),
        "Multinomial Naive Bayes (MNB) [BoW]": (MultinomialNB(), X_train_bow, X_test_bow),
        "Logistic Regression (LR) [BoW]": (LogisticRegression(max_iter=200, random_state=random_state), X_train_bow, X_test_bow),
    }
    
    results = []
    print("\n" + "="*80)
    print("EMPIRICAL EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    for name, (model, tr_x, te_x) in models.items():
        t0 = time.time()
        model.fit(tr_x, y_train)
        preds = model.predict(te_x)
        latency = (time.time() - t0) / len(y_test) * 1000
        
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, preds)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Latency (ms)": latency
        })
        
        print(f"\n---> {name}")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
    print("\n" + "="*80)
    return pd.DataFrame(results)

if __name__ == "__main__":
    import os
    
    dataset_path = "train.csv"
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading local dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
    else:
        print(f"[NOTICE] Local dataset '{dataset_path}' not found. Initializing benchmark synthetic pipeline...")
        df = generate_synthetic_benchmark_data(num_samples=2000)
        
    df = preprocess_text(df)
    results_df = run_evaluation_pipeline(df['cleaned_content'], df['label'])
    print("\nPipeline execution completed successfully.")