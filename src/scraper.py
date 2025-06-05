from google_play_scraper import Sort, reviews
import pandas as pd
from datetime import datetime
import os
import time
from urllib.error import URLError
import random
import socket

# Bank app IDs
BANK_APPS = {
    'CBE': 'com.combanketh.mobilebanking',
    'Dashen': 'com.dashen.dashensuperapp',
    'BOA': 'com.boa.boaMobileBanking'
    
}

def scrape_reviews(app_id, count=400, max_retries=5, initial_delay=10):
    socket.setdefaulttimeout(30)  # 30 seconds timeout
    for attempt in range(max_retries):
        try:
            result, continuation_token = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=count
            )
            
            # If we need more reviews, continue scraping
            while len(result) < count and continuation_token is not None:
                time.sleep(random.uniform(1, 3))  # Random delay between requests
                next_reviews, continuation_token = reviews(
                    app_id,
                    continuation_token=continuation_token
                )
                result.extend(next_reviews)
            
            return result
            
        except (URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {initial_delay} seconds...")
                time.sleep(initial_delay)
                initial_delay *= 1.5  # Exponential backoff
            else:
                print(f"Failed to scrape reviews after {max_retries} attempts")
                raise
#process the scraped reviews into a pandas DataFrame
def process_reviews(reviews_data, bank_name):
    processed_reviews = []
    
    for review in reviews_data:
        processed_reviews.append({
            'review': review['content'],
            'rating': review['score'],
            'date': review['at'].strftime('%Y-%m-%d'),
            'bank': bank_name
        })
    
    return pd.DataFrame(processed_reviews)

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    all_reviews = []
    
    # Scrape reviews for each bank
    for bank_name, app_id in BANK_APPS.items():
        print(f"\nScraping reviews for {bank_name}...")
        try:
            reviews_data = scrape_reviews(app_id)
            df = process_reviews(reviews_data, bank_name)
            all_reviews.append(df)
            
            # Save individual bank reviews
            df.to_csv(f'data/raw/{bank_name.lower()}_reviews.csv', index=False)
            print(f"Successfully saved {len(df)} reviews for {bank_name}")
            
            # Add delay between banks to avoid rate limiting
            time.sleep(random.uniform(5, 8))
            
        except Exception as e:
            print(f"Error scraping {bank_name}: {str(e)}")
            continue
    
    if all_reviews:
        # Combine all reviews
        combined_df = pd.concat(all_reviews, ignore_index=True)
        combined_df.to_csv('data/raw/all_bank_reviews.csv', index=False)
        print(f"\nTotal reviews collected: {len(combined_df)}")
    else:
        print("\nNo reviews were collected successfully.")

if __name__ == "__main__":
    main() 