import gradio as gr
import langdetect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import re

# Initialize analyzers
print("Initializing sentiment analyzer and translator...")
vader_analyzer = SentimentIntensityAnalyzer()
translator = GoogleTranslator(source='auto', target='en')

def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "unknown"

def translate_to_english(text, source_lang="auto"):
    try:
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def analyze_sentiment_with_vader(text):
    scores = vader_analyzer.polarity_scores(text)
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
    
    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "score": compound,
        "details": scores
    }

def analyze_sentiment(text):
    if not text:
        return "Please enter some text to analyze."
    
    # Detect language
    lang = detect_language(text)
    original_lang = lang
    
    # Store original text
    original_text = text
    
    # Translate if not English
    translated_text = None
    if lang != "en":
        translated_text = translate_to_english(text, lang)
        analysis_text = translated_text
    else:
        analysis_text = text
    
    # Analyze sentiment using VADER on English or translated text
    result = analyze_sentiment_with_vader(analysis_text)
    
    # Format the output
    output = ""
    if original_lang != "en":
        if original_lang == "id":
            lang_name = "Indonesian"
        else:
            lang_name = f"detected as {original_lang}"
        
        output += f"Detected Language: {lang_name}\n"
        output += f"Original Text: {original_text}\n"
        output += f"Translated to English: {translated_text}\n\n"
    else:
        output += "Detected Language: English\n\n"
    
    output += f"Sentiment: {result['sentiment']} {result['emoji']}\n"
    output += f"Compound Score: {result['score']:.2f}\n"
    output += f"Positive: {result['details']['pos']:.2f}, Negative: {result['details']['neg']:.2f}, Neutral: {result['details']['neu']:.2f}\n\n"
    
    # Add sentence-by-sentence analysis for longer texts
    sentences = re.split(r'[.!?]+', original_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) > 1:
        output += "Sentence-by-sentence analysis:\n"
        for sentence in sentences:
            if not sentence:
                continue
                
            # Detect language for this sentence
            sent_lang = detect_language(sentence)
            
            # Translate if not English
            if sent_lang != "en":
                translated_sentence = translate_to_english(sentence, sent_lang)
                sent_result = analyze_sentiment_with_vader(translated_sentence)
                output += f"• \"{sentence}\"\n"
                output += f"  Translated: \"{translated_sentence}\"\n"
                output += f"  Sentiment: {sent_result['sentiment']} ({sent_result['score']:.2f})\n"
            else:
                sent_result = analyze_sentiment_with_vader(sentence)
                output += f"• \"{sentence}\"\n"
                output += f"  Sentiment: {sent_result['sentiment']} ({sent_result['score']:.2f})\n"
    
    return output

# Create the Gradio interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(placeholder="Enter text in English, Indonesian, or any language...", lines=5),
    outputs=gr.Textbox(),
    title="Multilingual Sentiment Analysis with Translation",
    description="Analyze sentiment in any language by translating to English first, then using VADER.",
    examples=[
        ["I love this product! It's amazing and works perfectly."],
        ["Saya sangat senang dengan layanan ini. Terima kasih!"],
        ["This is terrible. Saya tidak suka dengan produk ini sama sekali."],
        ["The food was okay, but pelayanannya sangat lambat dan tidak ramah."],
        ["Harga produk ini terlalu mahal untuk kualitasnya yang biasa saja."]
    ]
)

# Launch the app
if __name__ == "__main__":
    print("Starting translation-based sentiment analysis app...")
    demo.launch()
    print("Sentiment analysis app is running!")