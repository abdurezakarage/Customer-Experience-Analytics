import pandas as pd
import numpy as np
from datetime import datetime
import os


class Preprocessor:
    def load_data(self, file_path):
        return pd.read_csv(file_path)
    #check duplicates
    def duplicate_check(self, df):
        return df.duplicated().sum()

    def remove_duplicates(self, df):
        
        return df.drop_duplicates(subset=['review', 'date', 'bank'], keep='first')
#check missing values
    def missing_values(self, df):
        return df.isnull().sum()


    def handle_missing_data(self, df):
      
        # Fill missing reviews with empty string
        df['review'] = df['review'].fillna('')
        
        # Fill missing ratings with median rating for that bank
        df['rating'] = df.groupby('bank')['rating'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Fill missing dates with current date
        df['date'] = df['date'].fillna(datetime.now().strftime('%Y-%m-%d'))
        
        return df

    def normalize_dates(self, df):
        """
        Ensure all dates are in YYYY-MM-DD format.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with normalized dates
        """
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df

def main():
    # Create processed data directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = Preprocessor()
    
    # Load the raw data
    print("Loading data...")
    df = preprocessor.load_data('data/raw/all_bank_reviews.csv')
    
    # Print initial statistics
    print(f"\nInitial number of reviews: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Check for duplicates
    print("Total number of rows:", len(df))
    print("\nNumber of duplicates based on all columns:", df.duplicated().sum())
    print("\nNumber of duplicates based on review, date, and bank:", df.duplicated(subset=['review', 'date', 'bank']).sum())

    # Show some examples of duplicates
    duplicates = df[df.duplicated(subset=['review', 'date', 'bank'], keep=False)]
    print("\nExample of duplicate entries:")
    print(duplicates.sort_values(['review', 'date', 'bank']).head(10))
    
    # Preprocess the data
    print("\nPreprocessing data...")
    df = preprocessor.remove_duplicates(df)
    print(f"Reviews after removing duplicates: {len(df)}")
    
    df = preprocessor.handle_missing_data(df)
    print("\nMissing values after handling:")
    print(df.isnull().sum())
    
    df = preprocessor.normalize_dates(df)
    
    # Save processed data
    output_path = 'data/processed/processed_reviews.csv'
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    # Print final statistics
    print("\nFinal statistics:")
    print(f"Total reviews: {len(df)}")
    print("\nReviews per bank:")
    print(df['bank'].value_counts())
    print("\nRating distribution:")
    print(df['rating'].value_counts().sort_index())

if __name__ == "__main__":
    main() 