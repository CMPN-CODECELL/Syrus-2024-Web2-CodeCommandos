from flask import Flask, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()  
    text = data.get('text', '')  

    if not text:
        return jsonify({'error': 'No text message provided'}), 400

    result = analyze_text_sentiment(text)
    return jsonify({'sentiment': result}), 200

def analyze_text_sentiment(text):
    polarity_scores = sia.polarity_scores(text)
    compound_score = polarity_scores['compound']

    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

if __name__ == '__main__':
    app.run(debug=True)
