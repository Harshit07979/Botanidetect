import numpy as np
import nltk
import string
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')  # tokenizer
nltk.download('wordnet')  # dictionary

lemmer = nltk.stem.WordNetLemmatizer()
stop_words = set(word.lower() for word in nltk.corpus.stopwords.words('english'))

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmer.lemmatize(token) for token in tokens]
    tokens = [token.lower() for token in tokens if token not in stop_words]
    return tokens

greet_input = ("hi", "hello", "whassup", "how are you?")
greet_res = ("Hi", "Hey", "Hey there!")

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_input:
            return random.choice(greet_res)

responses = {}

# Load existing responses from a JSON file
try:
    with open("responses.json", "r", encoding='utf-8') as file:
        responses = json.load(file)
except FileNotFoundError:
    pass

def response(msg):
    msg = msg.lower()
    if msg in responses:
        return responses[msg]
    else:
        # Check if the user's message is a greeting
        if greet(msg) is not None:
            return greet(msg)
        else:
            # Extract stored questions for comparison
            stored_questions = list(responses.keys())
            if len(stored_questions) > 0:
                # Use TfidfVectorizer to transform text into numerical representations
                tfidf_vectorizer = TfidfVectorizer(tokenizer=LemTokens, stop_words=None)
                tfidf_matrix = tfidf_vectorizer.fit_transform([msg] + stored_questions)
                cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
                
                # Find the most similar question
                most_similar_index = cosine_similarities.argmax()
                if cosine_similarities[0][most_similar_index] > 0.8:
                    return responses[stored_questions[most_similar_index]]
                
            answer = input("Bot: I don't know the answer. Please provide a response: ")
            responses[msg] = answer
            # Save updated responses to the JSON file
            with open("responses.json", "w", encoding='utf-8') as file:
                json.dump(responses, file, ensure_ascii=False, indent=4)
            return "Bot: Thank you for the information."

flag = True
print("Hello! I am the chatbot. Start typing your queries. I am here to help you!!")
while flag:
    msg = input("You: ")
    if msg.lower() == 'bye':
        flag = False
        print("Bot: Goodbye!")
    else:
        reply = response(msg)
        print("Bot:", reply)
