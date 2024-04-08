import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random


model = load_model('model.h5')
intents = json.loads(open('data.json', encoding="utf8").read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))


lemmatizer = WordNetLemmatizer()


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


def predict_class(sentence, model):
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

def getResponse(ints, intents_json):

    list_of_intents = intents_json['intents']

    if ints:
        tag = ints[0]['intent']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        # Access the last intent in the list
        last_intent = list_of_intents[-1]

        result = random.choice(last_intent['responses'])
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def getTags(category):
    int = intents['intents']

    result = []
     
    for i in int:
        if category in i['tag']:
            result.append(i['patterns'][0])
    return result

from flask import Flask, render_template, request


app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


@app.route("/getTag")
def get_tags():
    category = request.args.get('cat')
    return getTags(category)