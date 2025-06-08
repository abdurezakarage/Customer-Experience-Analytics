import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import re
from collections import defaultdict

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Define banking-specific theme categories and their keywords
        self.theme_categories = {
            'Account Access & Security': [
                'login', 'password', 'security', 'authentication', 'access', 'lock', 'unlock',
                'verification', 'biometric', 'fingerprint', 'face', 'id', 'secure'
            ],
            'Transaction & Payment': [
                'transfer', 'payment', 'transaction', 'send', 'receive', 'money', 'deposit',
                'withdraw', 'balance', 'fee', 'charge', 'cost', 'amount'
            ],
            'User Interface & Experience': [
                'app', 'interface', 'design', 'layout', 'screen', 'button', 'menu', 'navigation',
                'feature', 'function', 'option', 'setting', 'preference'
            ],
            'Customer Support & Service': [
                'support', 'service', 'help', 'assist', 'contact', 'response', 'wait', 'time',
                'agent', 'representative', 'chat', 'call', 'email'
            ],
            'Technical Performance': [
                'crash', 'error', 'bug', 'glitch', 'slow', 'lag', 'freeze', 'load', 'performance',
                'speed', 'reliability', 'stability', 'update'
            ]
        }
        
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def get_sentiment(self, text):
        """Get sentiment using DistilBERT"""
        # Handle empty or invalid text
        if not isinstance(text, str) or not text.strip():
            return 'neutral'
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                sentiment_score = scores[0][1].item()  # Probability of positive sentiment
                
            # Convert to sentiment label based on probability threshold
            if sentiment_score > 0.6:
                return 'positive'
            elif sentiment_score < 0.4:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return 'neutral'
 
    
    def extract_keywords(self, texts, n_keywords=50):
        """Extract keywords using TF-IDF"""
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list or tuple of strings")
        
        if not texts:
            raise ValueError("texts list cannot be empty")
            
        # Vectorize the dataset
        vectorizer = TfidfVectorizer(max_features=n_keywords)
        X = vectorizer.fit_transform(texts)
        
        # Get top keywords
        keywords = vectorizer.get_feature_names_out()
        print("Top Keywords:", keywords)
        
        return keywords
    
    def cluster_themes(self, texts, n_clusters=5):
        """Cluster texts into themes using K-means"""
        if not texts:
            return [], {}
            
        # Vectorize the texts
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        
        # Convert to dense array for KMeans
        X_dense = X.toarray()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_dense)
        
        # Extract top keywords from k-means cluster centers
        feature_names = vectorizer.get_feature_names_out()
        cluster_keywords = {}
        
        for i in range(n_clusters):
            center = kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-10:][::-1]
            cluster_keywords[i] = [feature_names[idx] for idx in top_indices]
        
        return clusters, cluster_keywords

    def map_to_theme_categories(self, keywords):
        """Map keywords to predefined banking theme categories"""
        theme_scores = defaultdict(int)
        
        for keyword in keywords:
            for theme, theme_keywords in self.theme_categories.items():
                if keyword in theme_keywords:
                    theme_scores[theme] += 1
        
        # Return the theme with the highest score, or 'Other' if no match
        if theme_scores:
            return max(theme_scores.items(), key=lambda x: x[1])[0]
        return 'Other'

    def analyze_reviews(self, df):
        """Main analysis pipeline"""
        # Preprocess texts
        df['processed_text'] = df['review'].apply(self.preprocess_text)
        
        # Get sentiment scores
        sentiment_results = df['review'].apply(self.get_sentiment)
        df['sentiment_label'] = sentiment_results
        
        # Extract keywords
        keywords = self.extract_keywords(df['processed_text'].tolist())
        df['keywords'] = [keywords] * len(df)
        
        # Cluster into themes and map to categories
        clusters, theme_keywords = self.cluster_themes(df['processed_text'].tolist())
        df['theme_cluster'] = clusters
        
        # Map clusters to banking-specific themes
        df['theme_category'] = df['theme_cluster'].apply(
            lambda x: self.map_to_theme_categories(theme_keywords[x])
        )
        
        # Calculate sentiment scores
        df['sentiment_score'] = df['review'].apply(
            lambda x: self.get_sentiment_score(x) if isinstance(x, str) else 0.5
        )
        
        # Save results with enhanced columns
        output_path = 'data/processed/sentiment_analysis_results.csv'
        df.to_csv(output_path, index=False)
        
        return df, theme_keywords
    
    def get_sentiment_score(self, text):
        """Get numerical sentiment score"""
        if not isinstance(text, str) or not text.strip():
            return 0.5
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                return scores[0][1].item()  # Probability of positive sentiment
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return 0.5

    def generate_summary(self, df):
        """Generate enhanced summary statistics"""
        summary = {
            'total_reviews': len(df),
            'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
            'mean_sentiment_by_bank': df.groupby('bank')['rating'].mean().to_dict(),
            'theme_distribution': df['theme_category'].value_counts().to_dict(),
            'theme_by_bank': df.groupby(['bank', 'theme_category']).size().unstack(fill_value=0).to_dict(),
            'mean_sentiment_by_theme': df.groupby('theme_category')['sentiment_score'].mean().to_dict()
        }
        return summary

    def plot_sentiment_distribution(self, df):
        """Plot sentiment distribution pie chart"""
        plt.figure(figsize=(8, 6))
        sentiment_counts = df['sentiment_label'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Distribution')
        plt.show()

    def plot_bank_sentiment(self, df):
        """Plot mean sentiment by bank"""
        plt.figure(figsize=(10, 6))
        bank_sentiment = df.groupby('bank')['rating'].mean()
        bank_sentiment.plot(kind='bar')
        plt.title('Mean Sentiment by Bank')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_keyword_cloud(self, df):
        """Generate word cloud from keywords"""
        plt.figure(figsize=(10, 6))
        all_keywords = ' '.join([' '.join(keywords) for keywords in df['keywords']])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Keywords')
        plt.show()

    def plot_sentiment_histogram(self, df):
        """Plot sentiment score distribution"""
        plt.figure(figsize=(8, 6))
        plt.hist(df['rating'], bins=20)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.show()

    def plot_theme_keywords(self, theme_keywords):
        """Plot top keywords for each theme cluster"""
        n_clusters = len(theme_keywords)
        fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 4*n_clusters))
        if n_clusters == 1:
            axes = [axes]
        
        for i, (cluster, keywords) in enumerate(theme_keywords.items()):
            axes[i].bar(range(len(keywords)), [1]*len(keywords))
            axes[i].set_xticks(range(len(keywords)))
            axes[i].set_xticklabels(keywords, rotation=45)
            axes[i].set_title(f'Theme Cluster {cluster+1} Keywords')
            axes[i].set_yticks([])
        
        plt.tight_layout()
        plt.show()

    def plot_theme_distribution(self, df):
        """Plot theme distribution by bank"""
        plt.figure(figsize=(12, 6))
        theme_by_bank = df.groupby(['bank', 'theme_category']).size().unstack(fill_value=0)
        theme_by_bank.plot(kind='bar', stacked=True)
        plt.title('Theme Distribution by Bank')
        plt.xlabel('Bank')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        plt.legend(title='Theme Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        