from flask import Flask, render_template, request, jsonify
import stockbot  # Ensure stockbot.py is in the same directory

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    ticker = request.args.get('ticker')
    if not ticker:
        return "No ticker provided.", 400
    
    # Use functions from stockbot.py to fetch and process data
    try:
        yahoo_headlines = stockbot.fetch_yahoo_news(ticker)
        finnhub_headlines = stockbot.fetch_finnhub_news(ticker)
        all_headlines = list(set(yahoo_headlines + finnhub_headlines))
        
        if not all_headlines:
            return "No headlines found.", 404
        
        sentiment_df = stockbot.analyze_sentiment(all_headlines)
        historical_data = stockbot.get_historical_data()
        model, vectorizer, tfidf_transformer = stockbot.train_model(historical_data)
        predictions = stockbot.predict_action(model, vectorizer, tfidf_transformer, all_headlines)
        
        results = [{"headline": headline, "action": action} for headline, action in zip(all_headlines, predictions)]
        return jsonify(results)
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
