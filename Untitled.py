#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

# Sample dataset (Can be expanded with more FAQs)
data = {
    "hi": "Hello! How can I assist you today?",
    "how are you": "I'm just a bot, but I'm doing great! How about you?",
    "what is AI": "AI stands for Artificial Intelligence, which enables machines to mimic human intelligence.",
    "bye": "Goodbye! Have a great day!"
}

# Load or update knowledge base
def load_knowledge_base():
    try:
        with open("chatbot_data.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return data

def save_knowledge_base(data):
    with open("chatbot_data.json", "w") as file:
        json.dump(data, file)

knowledge_base = load_knowledge_base()

# Train a simple model
def train_chatbot(data):
    X_train = list(data.keys())
    y_train = list(data.values())
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

model = train_chatbot(knowledge_base)

# Chatbot response function
def chatbot_response(user_input):
    user_input = user_input.lower()
    tokens = word_tokenize(user_input)
    
    for key in knowledge_base.keys():
        if key in user_input:
            return knowledge_base[key]
    
    # Try predicting response
    try:
        response = model.predict([user_input])[0]
    except:
        response = None
    
    if response:
        return response
    else:
        new_response = input("I don't know how to respond. How should I reply? ")
        knowledge_base[user_input] = new_response
        save_knowledge_base(knowledge_base)
        return "Got it! I'll remember that for next time."

# Running the chatbot
if __name__ == "__main__":
    print("Chatbot: Hello! Ask me anything. Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(user_input))


# In[ ]:


hi

