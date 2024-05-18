import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import psycopg2

# Import the necessary module
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

# Access environment variables as if they came from the actual environment
SECRET_KEY = os.getenv('SECRET_KEY')
SECRET_CODE = os.getenv('SECRET_CODE')
ADMIN_PASS=os.getenv('ADMIN_PASS')
ADMIN_EMAIL=os.getenv('ADMIN_EMAIL')
DB_PASS=os.getenv('DB_PASS')


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

def connect_db():
    return psycopg2.connect(
    database="intents_db", 
    user='postgres', 
    password=DB_PASS, 
    host='127.0.0.1', 
    port= '5432'
    )

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
    conn = connect_db()
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    cursor.execute('''SELECT tag, responses FROM intents''')

    rows = cursor.fetchall()
    list_of_intents = rows #[[tag, [responses]], [[tag, [responses]]...]

    conn.close()
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
    if msg == SECRET_CODE:
        res = "ADMIN"
        return res
    else:
        ints = predict_class(msg)
        print(ints)
        res = getResponse(ints)
        return res


def getTags(category):
    #Establishing the connection
    conn = connect_db()
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    cursor.execute('''SELECT tag, patterns FROM intents''')

    rows = cursor.fetchall()
    intents = rows

    result = []
    
    conn.close()
    for i in intents:
        if category in i[0]: #check if cat is in tag
            result.append(i[1][0]) #append first pattern
    return result

def getIntents():
    #Establishing the connection
    conn = connect_db()
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    cursor.execute('''SELECT tag, patterns, responses FROM intents''')

    rows = cursor.fetchall()

    result = rows
    conn.close()
    return result


from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
app.static_folder = 'static'
app.secret_key = SECRET_KEY


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Dummy user validation
        if username == ADMIN_EMAIL and password == ADMIN_PASS:
            session['username'] = username
            return redirect(url_for('admin'))
        else:
            return render_template('admin_login.html')
    else:
        return render_template('admin_login.html')
    
@app.route('/logout')
def logout():
    session["name"] = None
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/admin')
def admin():
    if not session.get("username"):
        # if not there in the session then redirect to the login page
        return redirect("/login")
    if 'username' in session:
        username = session['username']
        patterns=getTags('')

        int_res = []
        pat_res = []
        intents=getIntents()
        for i in range(0,len(intents)):
            if "MISC" not in intents[i][0]:
                int_res.append(intents[i])
                pat_res.append(patterns[i])
        return render_template('admin.html', len=len(pat_res), patterns=pat_res, intents=int_res)


@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')

@app.route("/getReply")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route("/update")
def update_response():
    pattern = request.args.get('pattern')
    oldResponse = request.args.get('old')
    newResponse = request.args.get('new')

    conn = connect_db()
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    cursor.execute(f"""SELECT tag, patterns, responses FROM intents WHERE '{pattern}' = ANY(patterns)""")

    rows = cursor.fetchone()

    # print(rows)
    if pattern in rows[1]: #check if cat is in tag
        for count, i in enumerate(rows[2]):
            if oldResponse == i:
                rows[2][count]=newResponse
                cursor.execute(f"""UPDATE intents SET responses[{count+1}]='{newResponse}' WHERE tag='{rows[0]}'""")
                conn.commit()
                break
    else:
        conn.close()
        return "Update FAILED"

    conn.close()
    return f"Updated Successfully"

@app.route("/add")
def add_intent():
    tag = request.args.get('tag')
    patterns = request.args.get('pattern')
    responses = request.args.get('response')

    conn = connect_db()
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    
    cursor.execute(f'''INSERT INTO intents(tag, patterns, responses) VALUES ('{tag}',  ARRAY['{patterns}'],  ARRAY['{responses}'])''')
    conn.commit()
    conn.close()
    return f"Added Successfully. Do not forget to retrain the model."

@app.route("/delete")
def delete_intent():
    patterns = request.args.get('pattern')

    conn = connect_db()
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    
    cursor.execute(f'''DELETE FROM intents WHERE '{patterns}' = ANY(patterns)''')
    conn.commit()
    conn.close()
    return f"Deleted Successfully"

@app.route("/getTag")
def get_tags():
    category = request.args.get('cat')
    return getTags(category)