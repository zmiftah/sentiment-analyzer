import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import gradio as gr


def create_sentiment_analyzer():
    """Create and return a sentiment analysis pipeline."""
    # Load pre-trained model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Create sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_analyzer


def analyze_sentiment(text, analyzer):
    """Analyze the sentiment of the given text."""
    result = analyzer(text)
    return result[0]


def run_examples():
    """Run sentiment analysis on example texts."""
    print("Loading sentiment analysis model...")
    sentiment_analyzer = create_sentiment_analyzer()
    print("Model loaded successfully!")

    # Example texts
    example_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Terrible service.",
        "The movie was okay. Not great, not terrible.",
        "I'm feeling quite neutral about the whole situation."
    ]

    print("\n--- Example Sentiment Analysis ---")
    for text in example_texts:
        result = analyze_sentiment(text, sentiment_analyzer)
        sentiment = result['label']
        confidence = result['score']
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")


def run_interactive():
    """Run interactive sentiment analysis in the console."""
    sentiment_analyzer = create_sentiment_analyzer()

    print("\n--- Interactive Mode ---")
    print("Enter text to analyze (or 'quit' to exit):")

    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() == 'quit':
            break

        if user_input.strip():
            result = analyze_sentiment(user_input, sentiment_analyzer)
            sentiment = result['label']
            confidence = result['score']
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.4f}")
        else:
            print("Please enter some text to analyze.")

    print("Thank you for using the sentiment analyzer!")


def run_gradio_interface():
    """Run the Gradio web interface."""
    # Create the sentiment analyzer
    sentiment_analyzer = create_sentiment_analyzer()

    def predict(text):
        """Analyze sentiment and format the result for Gradio."""
        if not text.strip():
            return "Please enter some text to analyze."

        result = analyze_sentiment(text, sentiment_analyzer)
        sentiment = result['label']
        confidence = result['score']

        return f"Sentiment: {sentiment} (Confidence: {confidence:.4f})"

    # Create Gradio interface
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(placeholder="Enter text to analyze..."),
        outputs="text",
        title="Sentiment Analysis",
        description="Analyze the sentiment of text using a pre-trained DistilBERT model."
    )

    # Launch the app
    demo.launch()


if __name__ == "__main__":
    # Uncomment one of these based on how you want to run the app
    # run_examples()  # Run with example texts
    # run_interactive()  # Run in interactive console mode
    run_gradio_interface()  # Run with Gradio web interface
