#MAKE SURE TO pip install flask yfinance requests finnhub nltk scikit-learn pandas

import nltk
nltk.download('vader_lexicon')

import yfinance as yf
import finnhub
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import random
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import class_weight

# Download VADER Lexicon
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Initialize Finnhub API
FINNHUB_API_KEY = "ctgt59hr01qg92n2pkv0ctgt59hr01qg92n2pkvg"  # Replace with your Finnhub API key
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Step 1: Fetch news headlines from Yahoo Finance
def fetch_yahoo_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        headlines = [item['title'] for item in news] if news else []
        return headlines
    except Exception as e:
        print(f"Error fetching Yahoo Finance news: {e}")
        return []

# Step 2: Fetch news headlines from Finnhub
def fetch_finnhub_news(ticker):
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        news = finnhub_client.company_news(ticker, _from=yesterday, to=today)
        headlines = [item['headline'] for item in news]
        return headlines
    except Exception as e:
        print(f"Error fetching Finnhub news: {e}")
        return []

# Step 3: Analyze sentiment and extract scores
def analyze_sentiment(headlines):
    data = []
    for headline in headlines:
        scores = sia.polarity_scores(headline)
        data.append({
            'headline': headline,
            'pos': scores['pos'],
            'neg': scores['neg'],
            'neu': scores['neu'],
            'compound': scores['compound']
        })
    return pd.DataFrame(data)

# Step 4: Simulated historical data for training
def get_historical_data():
# Simulated headlines for 100 entries
    data = {
    'headline': [
        "Stock prices hit a new high today",
        "Company reports lower revenue this quarter",
        "The market is showing strong growth",
        "Company faces lawsuits, shares drop",
        "Earnings report exceeds expectations",
        "Revenue falls short, stock price plunges",
        "Market hits all-time high due to demand surge",
        "Company loses market share to competitors",
        "Stocks rise as inflation concerns ease",
        "Stock prices plummet due to weak guidance",
        "Company announces new product launch",
        "Stock price stabilizes after recent drop",
        "Analysts predict strong earnings growth",
        "Concerns over global economy impact stock prices",
        "Stock prices hit a new high today",
        "Company reports lower revenue this quarter",
        "The market is showing strong growth",
        "Company faces lawsuits, shares drop",
        "Earnings report exceeds expectations",
        "Revenue falls short, stock price plunges",
        "Company XYZ Faces Lawsuit for Misleading Financial Statements",  # sell example
        "Company XYZ Reports Loss of Market Share Amid Intense Competition",  # sell example
        "Stock Price Plummets After Company Reports Quarterly Losses",  # sell example
        "Shares soar after positive earnings report",
        "Company announces major acquisition, stock jumps",
        "Stock price dips despite strong fundamentals",
        "Market rallies after inflation fears ease",
        "Company sets ambitious growth targets, stock jumps",
        "Strong demand for new product pushes shares higher",
        "Negative outlook from analysts leads to price drop",
        "Company's revenue growth slows down, stock drops",
        "Weak earnings report causes market selloff",
        "Strong performance in international markets boosts stock",
        "Shares fall due to CEO resignation",
        "Stock price climbs after favorable regulatory news",
        "Difficult market conditions hurt stock price",
        "Analysts lower target price for the stock",
        "Shares hit new low after disappointing results",
        "Company declares higher dividends, shares rise",
        "Stock remains steady despite economic uncertainty",
        "Company faces competition from new market entrant, stock dips",
        "Bulls are back as tech stocks surge",
        "Investors optimistic about upcoming product launch",
        "Lawsuit filed against company, stock drops",
        "Company announces expansion into new markets, stock soars",
        "Investor confidence boosts stock price",
        "Company misses earnings expectations, stock drops",
        "Company reports better-than-expected earnings",
        "Government subsidy for green energy pushes stocks higher",
        "Stock rebounds after recent market correction",
        "Concerns over new competition put pressure on stock",
        "Company focuses on cost-cutting, stock prices rise",
        "Weak sales result in lower stock price",
        "Stock continues to struggle after bad earnings call",
        "New CEO hired, investors react positively",
        "Company underperforms compared to competitors, stock drops",
        "Positive news from overseas markets pushes stock up",
        "Company announces restructuring, stock price drops",
        "Investors remain cautious despite positive earnings",
        "Market decline hits company's stock hard",
        "Stock climbs after positive analyst recommendation",
        "Investors sell off shares after weak guidance",
        "Company's stock rallies after surprise earnings beat",
        "Company faces legal troubles, stock tanks",
        "Analyst upgrades stock after strong earnings report",
        "Debt concerns hurt company stock",
        "Stock price drops as competition intensifies",
        "Stock hits new high after breakthrough product launch",
        "CEO optimistic about growth, stock price rises",
        "Investor sentiment shifts, stock price falls",
        "Stock price surges after positive quarterly report",
        "Company struggles with cash flow issues, stock dips",
        "Investors cautious as the market remains volatile",
        "Company hits a milestone, stock price rises",
        "Lack of product innovation causes stock price decline",
        "Company faces government scrutiny, stock drops",
        "Stock sees price drop after negative earnings report",
        "Analysts maintain neutral stance on stock",
        "Weak market conditions impact company's stock",
        "Stock rises on rumors of a merger",
        "High earnings report boosts stock price",
        "Company announces major restructuring, stock remains flat",
        "Tech stocks outperform, boosting market optimism",
        "Investor excitement pushes stock to new highs",
        "Stock prices stabilize after recent sell-off",
        "Company sees growth in new markets, stock price rises",
        "Lower-than-expected growth triggers sell-off",
        "Company beats earnings expectations, stock surges",
        "Unfavorable news from competitors hurt stock",
        "Company reduces forecast, stock price falls",
        "Global expansion boosts company stock",
        "Stock fluctuates as market conditions change",
        "Company's poor performance leads to stock decline",
        "Investors sell shares amid regulatory uncertainty",
        "Stock hits all-time high after strong earnings report",
        "Investors remain optimistic despite market volatility",
        "Stock drops after unexpected management changes",
        "Investor optimism sends stock prices higher",
        "Company faces tough competition, stock price weakens",
        "Stock price surges after positive earnings surprise",
        "Company announces plans for growth, stock rallies",
        "Negative earnings report causes stock price to dip",
        "Stock experiences volatility amidst economic concerns",
        "Investors optimistic despite global challenges",
        "Company struggles to meet market expectations",
        "Company outperforms, stock price climbs",
        "Stock remains flat after earnings announcement",
        "Company announces significant market expansion",
        "Shares experience a slight decline after major announcement",
        "red",
        "green",
        "red",
        "green",
        "red",
        "green"
    ],
    'action': [
        'buy', 'sell', 'buy', 'sell', 'buy', 'sell', 'buy', 'sell', 'buy', 'sell', 'buy_more', 'hold', 'buy_more', 'neutral', 'buy', 'sell', 'buy', 'sell', 'buy', 'sell', 
        'sell', 'sell', 'buy', 'sell', 'buy', 'buy', 'sell', 'buy', 'buy', 'buy', 'sell', 'buy', 'buy', 'sell', 'buy', 'sell', 'neutral', 'sell', 'buy', 'buy', 'buy', 
        'neutral', 'buy', 'sell', 'sell', 'buy', 'sell', 'neutral', 'sell', 'buy', 'buy', 'sell', 'neutral', 'buy', 'buy', 'sell', 'sell', 'sell', 'buy', 'neutral', 
        'sell', 'buy', 'sell', 'neutral', 'sell', 'buy', 'sell', 'buy', 'buy', 'sell', 'buy', 'neutral', 'buy', 'neutral', 'sell', 'buy', 'sell', 'neutral', 'buy', 
        'neutral', 'buy', 'buy', 'sell', 'buy', 'neutral', 'neutral', 'buy', 'sell', 'buy', 'neutral', 'sell', 'buy', 'sell', 'neutral', 'neutral', 'buy', 'sell', 
        'neutral', 'sell', 'neutral', 'buy', 'buy', 'sell', 'buy', 'neutral', 'buy', 'neutral', 'sell', 'buy', 'sell', 'buy', 'sell', 'buy', 'sell', 'buy'
    ]
}
    return pd.DataFrame(data)

# Step 5: Train a logistic regression model
def train_model(data):
    # Check class distribution
    print("\nClass Distribution in Training Data:")
    print(data['action'].value_counts())

    # Vectorize the headlines (Convert text to features)
    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    X_counts = vectorizer.fit_transform(data['headline'])
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    # Target variable (action)
    y = data['action']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Compute class weights to handle imbalance
    class_weights = class_weight.compute_class_weight('balanced', classes=y.unique(), y=y)
    weights_dict = dict(zip(y.unique(), class_weights))

    # Train a logistic regression model with class weights
    model = LogisticRegression(class_weight=weights_dict, max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, vectorizer, tfidf_transformer

# Step 6: Predict action for new headlines
def predict_action(model, vectorizer, tfidf_transformer, headlines):
    X_counts = vectorizer.transform(headlines)
    X_tfidf = tfidf_transformer.transform(X_counts)
    predictions = model.predict(X_tfidf)
    return predictions

# Main Function
def main():
    ticker = input("Enter the stock ticker (e.g., AAPL, TSLA): ").strip().upper()

    # Step 1: Fetch news headlines
    yahoo_headlines = fetch_yahoo_news(ticker)
    finnhub_headlines = fetch_finnhub_news(ticker)
    all_headlines = list(set(yahoo_headlines + finnhub_headlines))

    if not all_headlines:
        print("No headlines found.")
        return

    print(f"\nFetched {len(all_headlines)} headlines.\n")

    # Step 2: Analyze sentiment (optional step to view scores)
    sentiment_df = analyze_sentiment(all_headlines)
    print("Sentiment Scores for Headlines:")
    print(sentiment_df[['headline', 'compound']])

    # Step 3: Get historical data and train the model
    historical_data = get_historical_data()
    model, vectorizer, tfidf_transformer = train_model(historical_data)

    # Step 4: Predict action for new headlines
    predictions = predict_action(model, vectorizer, tfidf_transformer, all_headlines)

    # Step 5: Display predictions
    print("\nPredictions for Latest Headlines:")
    for headline, prediction in zip(all_headlines, predictions):
        print(f"Headline: {headline}\nPredicted Action: {prediction}\n")

    # Step 6: Count actions and provide ultimate recommendation
    action_counts = pd.Series(predictions).value_counts()
    print("\nAction Counts:")
    print(action_counts)

    # Ultimate recommendation based on majority
    if action_counts.get('buy', 0) > action_counts.get('sell', 0) and action_counts.get('buy', 0) > action_counts.get('hold', 0):
        ultimate_recommendation = "Ultimate Recommendation: Buy"
    elif action_counts.get('sell', 0) > action_counts.get('buy', 0) and action_counts.get('sell', 0) > action_counts.get('hold', 0):
        ultimate_recommendation = "Ultimate Recommendation: Sell"
    else:
        ultimate_recommendation = "Ultimate Recommendation: Hold"

    print(ultimate_recommendation)

if __name__ == "__main__":
    main()
