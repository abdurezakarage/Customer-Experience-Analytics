# Customer Experience Analytics

This project analyzes customer reviews from banking applications on the Google Play Store to gain insights into customer satisfaction and experience.

## Project Structure

```
Customer-Experience-Analytics/
├── src/
│   ├── __init__.py
│   ├── scraper.py
│   └── preprocessor.py
├── data/
│   └── raw/
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Collection

The project collects reviews from three banking applications:
- Commercial Bank of Ethiopia (CBE)
- Bank of America (BOA)
- Dashen Bank

Each bank's reviews are collected using the Google Play Store Scraper, targeting 400+ reviews per bank.

## Data Preprocessing

The collected data undergoes the following preprocessing steps:
1. Removal of duplicate reviews
2. Handling of missing data
3. Date normalization to YYYY-MM-DD format
4. Data organization into a structured CSV format

## Output

The processed data is saved as a CSV file with the following columns:
- review: The text content of the review
- rating: Numerical rating (1-5)
- date: Review date in YYYY-MM-DD format
- bank: Name of the bank
- source: Source of the review (Google Play Store)

## Git Workflow

- Branch: task-1
- Regular commits with meaningful messages
- Commits are made after completing logical chunks of work