




	# <========================================================= Importing Required Libraries & Functions =================================================>
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import nltk
nltk.download('stopwords')
nltk.download("punkt")
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np
from keras.models import load_model
from PIL import Image
import random

# <========================================================== Load Assets =========================================================================>
model = load_model('chatbot_model.h5')
with open('intents3.json') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# <========================================================= Helper Functions =====================================================================>
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def chatbot_response(text):
    ints = predict_class(text, model)
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm not sure I understand. Can you rephrase?"

def generate_response(prompt):
    return chatbot_response(prompt)

# <========================================================= Streamlit UI Setup (Final Version) ===============================================================>
im = Image.open('boy.png')
st.set_page_config(layout="wide", page_title="Career Guidance ChatBot", page_icon=im)


# Header
st.markdown("""
    <div style="background: linear-gradient(to right, #4635B1, #B771E5); padding: 10px; border-radius: 10%">
        <h1 style="color: #AAB99A; font-size: 48px; font-weight: bold">
           <center> <span style="color: black; font-size: 64px">𝕸</span>𝖆𝖗𝖌<span style="color: black; font-size: 64px">𝕯</span>𝖆𝖗𝖘𝖍𝖆𝖓 </center>
        </h1>
    </div>
""", unsafe_allow_html=True)

# Basic styling
st.markdown("""
    <style>
        #MainMenu, footer {visibility: hidden;}
        .block-container {
            padding-bottom: 100px;
        }
        body {
            background-color: #FFFBCA; 
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title(' Career Guidance ChatBot')
    st.markdown('''
    ## About
    This app has been developed by  students of CGC Jhanjeri:

    - Kumar Shivendu [2338465]  
    - Aayush Thakur [2338382]  
    - Ayush kumar [2338426]  
    - Abhay singh rawat [2338385]  
    ''')
    add_vertical_space(5)

# Session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm an AI Career Counselor, how may I help you?"]
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']
if 'input' not in st.session_state:
    st.session_state['input'] = ""

# Scrollable chat window
st.markdown("---")
with st.container():
    st.markdown("<div style='height:140px; overflow-y:auto; padding:10px'>", unsafe_allow_html=True)
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=f"user_{i}")
        message(st.session_state['generated'][i], key=f"bot_{i}")
    st.markdown("</div>", unsafe_allow_html=True)

# Input box and send button
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("You:", value=st.session_state["input"], key="input_box", label_visibility="collapsed", placeholder="Type your message here...")

with col2:
    send_clicked = st.button("Send")

# Logic: send on button click or enter
if send_clicked or (user_input and user_input != st.session_state["input"]):
    # Ensure the input is updated immediately before adding to the past list
    st.session_state.past.append(user_input)
    response = generate_response(user_input)
    st.session_state.generated.append(response)
    st.session_state["input"] = ""  # Clear the input field

   


