import gradio as gr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    if not text:
        return "Please enter some text to analyze."
    
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    scores = analyzer.polarity_scores(text)
    
    # Determine overall sentiment
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "Positive"
        emoji = "😊"
    elif compound <= -0.05:
        sentiment = "Negative"
        emoji = "😞"
    else:
        sentiment = "Neutral"
        emoji = "😐"
    
    # Format the result
    result = f"Sentiment: {sentiment} {emoji}\n"
    result += f"Compound Score: {compound:.2f}\n"
    result += f"Positive: {scores['pos']:.2f}, Negative: {scores['neg']:.2f}, Neutral: {scores['neu']:.2f}"
    
    # Add sentence-by-sentence analysis for longer texts
    sentences = text.split('.')
    if len(sentences) > 1:
        result += "\n\nSentence-by-sentence analysis:"
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Skip empty sentences
                sent_scores = analyzer.polarity_scores(sentence)
                sent_compound = sent_scores['compound']
                if sent_compound >= 0.05:
                    sent_sentiment = "Positive"
                elif sent_compound <= -0.05:
                    sent_sentiment = "Negative"
                else:
                    sent_sentiment = "Neutral"
                result += f"\n• \"{sentence}\": {sent_sentiment} ({sent_compound:.2f})"
    
    return result

# Create the Gradio interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(placeholder="Enter text to analyze sentiment...", lines=5),
    outputs=gr.Textbox(),
    title="Sentiment Analysis Tool (VADER)",
    description="Enter text to analyze its sentiment (positive, negative, or neutral) using VADER.",
    examples=[
        ["I love this product! It's amazing and works perfectly."],
        ["This is the worst experience I've ever had. Terrible customer service."],
        ["The weather today is cloudy with a chance of rain."],
        ["I'm happy about the good news but concerned about the potential challenges ahead."]
    ]
)

# Launch the app
if __name__ == "__main__":
    print("Starting VADER sentiment analysis app...")
    demo.launch()
    print("Sentiment analysis app is running!")