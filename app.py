import streamlit as st
import os
import numpy as np
import torch
from langdetect import detect
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Load environment variables
load_dotenv()

class MultilingualSentimentChatbot:
    def __init__(self):
        """
        Initialize Sentiment Analysis, Multilingual Configuration, and Groq LLM.
        """
        # Initialize Hugging Face Sentiment Model
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Initialize Groq LLM for response generation
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment variables.")

        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="mixtral-8x7b-32768"
        )

        # Initialize session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Rest of the initialization code remains the same
        self.LANGUAGES = {
            "en": {
                "name": "English",
                "system_prompt": "You are an intelligent, helpful assistant communicating in English. Provide clear, concise, and accurate responses.",
                "language_instruction": "Respond strictly in English."
            },
            "es": {
                "name": "Spanish",
                "system_prompt": "Eres un asistente inteligente y útil que se comunica en español. Proporciona respuestas claras, concisas y precisas.",
                "language_instruction": "Responde estrictamente en español."
            },
            "fr": {
                "name": "French",
                "system_prompt": "Vous êtes un assistant intelligent et utile qui communique en français. Fournissez des réponses claires, concises et précises.",
                "language_instruction": "Répondez strictement en français."
            },
            "hi": {
                "name": "Hindi",
                "system_prompt": "आप एक बुद्धिमान और सहायक सहायक हैं जो हिंदी में संवाद कर रहे हैं। स्पष्ट, संक्षिप्त और सटीक उत्तर प्रदान करें।",
                "language_instruction": "केवल हिंदी में उत्तर दें।"
            }
        }

        self.sentiment_intros = {
            'Positive': [
                "Great! Here's a helpful response:",
                "Wonderful! Let me help you with that:",
                "Sounds good! Here's what I think:"
            ],
            'Neutral': [
                "I'll help you with that:",
                "Let me provide some information:",
                "Here's a response to your query:"
            ],
            'Negative': [
                "I understand. Let me help you:",
                "I'm here to assist. Here's a response:",
                "I'll do my best to help:"
            ]
        }

    def analyze_sentiment(self, text):
        """Previous analyze_sentiment implementation remains the same"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_label_idx = probabilities.argmax().item()
        label_names = ['Negative', 'Neutral', 'Positive']
        sentiment = label_names[predicted_label_idx]
        confidence = probabilities[0][predicted_label_idx].item()
        return sentiment, confidence

    def detect_language(self, text: str) -> str:
        """Previous detect_language implementation remains the same"""
        try:
            detected_lang = detect(text)
            return detected_lang if detected_lang in self.LANGUAGES else "en"
        except Exception as e:
            logging.error(f"Language detection error: {str(e)}")
            return "en"

    def generate_multilingual_prompt(self, user_input: str, language: str, sentiment: str) -> str:
        """Generate a multilingual prompt with language-specific and sentiment-aware instructions."""
        lang_config = self.LANGUAGES.get(language, self.LANGUAGES["en"])
        
        # Get recent chat history
        recent_history = st.session_state.chat_history[-5:] if st.session_state.chat_history else []
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        multilingual_prompt = f"""
{lang_config['system_prompt']}

{lang_config['language_instruction']}

Context: The user's message has a {sentiment} sentiment.

Conversation History:
{history_str}

User Input: {user_input}

Respond carefully, maintaining the specified language, sentiment, and context.
"""
        return multilingual_prompt

    def generate_response(self, user_input: str, language: str, sentiment: str) -> str:
        """Generate a response using Groq LLM with language-specific and sentiment-aware prompting."""
        try:
            multilingual_prompt = self.generate_multilingual_prompt(user_input, language, sentiment)
            response = self.llm.invoke(multilingual_prompt)
            
            # Update session state chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Keep only the last 10 messages
            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = st.session_state.chat_history[-10:]
                
            return response.content
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an issue generating a response. Please try again."

    def select_sentiment_intro(self, sentiment):
        """Previous select_sentiment_intro implementation remains the same"""
        return np.random.choice(self.sentiment_intros[sentiment])

    def save_sentiment_analysis(self, user_input, sentiment, confidence):
        """Previous save_sentiment_analysis implementation remains the same"""
        try:
            with open("sentiment_analysis_log.txt", "a") as f:
                f.write(f"Input: {user_input}\nSentiment: {sentiment}\nConfidence: {confidence:.2%}\n\n")
            st.success("Sentiment analysis logged successfully!")
        except Exception as e:
            st.error(f"Error saving sentiment analysis: {e}")

    def display_chat_history(self):
        """Display the chat history using Streamlit's chat message containers"""
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def run_chatbot(self):
        """Main chatbot interface and logic with chat history display"""
        st.title("🌐 Multilingual Sentiment-Aware Chatbot")

        # Sidebar for language settings
        st.sidebar.header("Language Settings")
        selected_language = st.sidebar.selectbox(
            "Choose Language (or detect automatically)",
            options=["auto"] + list(self.LANGUAGES.keys()),
            format_func=lambda x: "Detect Automatically" if x == "auto" else self.LANGUAGES[x]["name"]
        )

        # Clear chat history button
        if st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

        # Sentiment logging toggle
        log_sentiment = st.sidebar.checkbox("Log Sentiment Analysis", value=False)

        # Display existing chat history
        self.display_chat_history()

        # Chat Input
        user_input = st.chat_input("Enter your message")

        if user_input:
            # Detect language if set to automatic detection
            detected_lang = self.detect_language(user_input)
            language_to_use = detected_lang if selected_language == "auto" else selected_language

            # Analyze Sentiment
            sentiment, confidence = self.analyze_sentiment(user_input)

            # Optional Sentiment Logging
            if log_sentiment:
                self.save_sentiment_analysis(user_input, sentiment, confidence)

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Display analysis results
            with st.expander("Message Analysis", expanded=False):
                st.write(f"🧠 Sentiment Detected: {sentiment}")
                st.write(f"🎯 Confidence: {confidence:.2%}")
                st.write(f"🌍 Language: {self.LANGUAGES.get(language_to_use, {}).get('name', 'Unknown')}")

            # Generate and display response
            try:
                sentiment_intro = self.select_sentiment_intro(sentiment)
                bot_response = self.generate_response(user_input, language_to_use, sentiment)

                with st.chat_message("assistant"):
                    st.markdown(f"{sentiment_intro}\n\n{bot_response}")

            except Exception as e:
                st.error(f"An error occurred while generating response: {e}")

def main():
    # Initialize and run the chatbot
    chatbot = MultilingualSentimentChatbot()
    chatbot.run_chatbot()

if __name__ == "__main__":
    main()