import gradio as gr
from textblob import TextBlob

def analyze_sentiment(text):
    if not text:
        return "Please enter some text to analyze."
    
    # Create a TextBlob object
    blob = TextBlob(text)
    
    # Get the sentiment polarity (range: -1 to 1)
    polarity = blob.sentiment.polarity
    
    # Determine sentiment category
    if polarity > 0.1:
        sentiment = "Positive"
        emoji = "😊"
    elif polarity < -0.1:
        sentiment = "Negative"
        emoji = "😞"
    else:
        sentiment = "Neutral"
        emoji = "😐"
    
    # Format the result
    result = f"Sentiment: {sentiment} {emoji}\nPolarity Score: {polarity:.2f}"
    
    # Add some example sentences from the text with their individual sentiment
    if len(text.split()) > 10:
        result += "\n\nSentence-by-sentence analysis:"
        for sentence in blob.sentences:
            sent_polarity = sentence.sentiment.polarity
            if sent_polarity > 0.1:
                sent_sentiment = "Positive"
            elif sent_polarity < -0.1:
                sent_sentiment = "Negative"
            else:
                sent_sentiment = "Neutral"
            result += f"\n• \"{sentence}\": {sent_sentiment} ({sent_polarity:.2f})"
    
    return result

# Create the Gradio interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(placeholder="Enter text to analyze sentiment...", lines=5),
    outputs=gr.Textbox(),
    title="Sentiment Analysis Tool",
    description="Enter text to analyze its sentiment (positive, negative, or neutral).",
    examples=[
        ["I love this product! It's amazing and works perfectly."],
        ["This is the worst experience I've ever had. Terrible customer service."],
        ["The weather today is cloudy with a chance of rain."],
        ["I'm happy about the good news but concerned about the potential challenges ahead."]
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
    print("Sentiment analysis app is running!")