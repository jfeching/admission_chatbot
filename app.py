import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import psycopg2

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

model = None
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))


lemmatizer = WordNetLemmatizer()

def load():
    global model
    # Load the TensorFlow model here
    model = load_model('model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag) 


def predict_class(sentence):
    global model
    if model is None:
        load()
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.7

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    print(return_list)
    return return_list

def getResponse(ints):
    conn = psycopg2.connect(
    database="intents_db", user='postgres', password='cisfran567', host='127.0.0.1', port= '5432'
    )
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    cursor.execute('''SELECT tag, responses FROM intents''')

    rows = cursor.fetchall()
    list_of_intents = rows #[[tag, [responses]], [[tag, [responses]]...]

    if ints:
        tag = ints[0]['intent']
        for i in list_of_intents:
            if i[0] == tag: #i[0] is the tag
                result = random.choice(i[1]).encode('latin-1').decode('unicode-escape') #i[1] is the list of responses
                break
    else:
        # Access the last intent in the list
        last_intent = rows[-1]

        result = random.choice(last_intent[1])
    return result

def chatbot_response(msg):
    ints = predict_class(msg)
    print(ints)
    res = getResponse(ints)
    return res


def getTags(category):
    #Establishing the connection
    conn = psycopg2.connect(
    database="intents_db", user='postgres', password='cisfran567', host='127.0.0.1', port= '5432'
    )
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    cursor.execute('''SELECT tag, patterns FROM intents''')

    rows = cursor.fetchall()
    int = rows

    result = []
     
    for i in int:
        if category in i[0]: #check if cat is in tag
            result.append(i[1][0]) #append first pattern
    return result

from flask import Flask, render_template, request


app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route("/getReply")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


@app.route("/getTag")
def get_tags():
    category = request.args.get('cat')
    return getTags(category)