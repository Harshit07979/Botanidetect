import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import urllib.request
import numpy as np
from PIL import Image
import nltk
import string
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')  # tokenizer
nltk.download('wordnet')  # dictionary

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

#configure for chatbot
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





app.secret_key= "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER  # Directory to store uploaded images
app.config["MAX_CONTENT_LENGTH"] = 16*1024*1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


weights1 = np.load("weights1.npy")
bias1 = np.load("bias1.npy")
weights2 = np.load("weights2.npy")
bias2 = np.load("bias2.npy")
# wp1= np.load("wp1.npy")
# wp2= np.load("wp2.npy")
# bp1= np.load("bp1.npy")
# bp2= np.load("bp2.npy")

img_size= 128
batch_size= 16
channels= 3
epochs= 16

path="potato2" #directory path of image set file

class_names= os.listdir(path)

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    ex= np.exp(x- np.max(x, axis=-1, keepdims=True))
    return ex/ ex.sum(axis=-1, keepdims=True)


@app.route('/')
def home():
    global jfile
    jfile=""
    return render_template("home.html")

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash("No image selected for uploading")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        
        pth = "static/uploads/" + filename
        img = Image.open(pth)
        img = img.resize((img_size, img_size))
        img = np.array(img) / 255.0

        input_size = img_size * img_size * channels
        
        # Perform the forward pass for the new image
        input_image = img.reshape(-1, input_size)
        hidden_layer = np.dot(input_image, weights1) + bias1
        hidden_activation = relu(hidden_layer)
        pred = np.dot(hidden_activation, weights2) + bias2
        pred = softmax(pred)

        # Interpret the predictions
        predicted_class = np.argmax(pred)
        predicted_class_name = class_names[predicted_class]
        print(predicted_class_name)
        global jfile
        if(predicted_class_name=="Potato___Early_Blight"):
            jfile+="responses.json"
        elif(predicted_class_name=="Potato___Late_Blight"):
            jfile+="lateres.json"
        elif(predicted_class_name=="Potato___Healthy"):
            jfile+="healres.json"
        print(jfile)
        flash("Image Successfully uploaded and displayed below")
        flash(predicted_class_name)
        return render_template("home.html", filename=filename, predicted_class_name=predicted_class_name,jfile=jfile)
        
    else:
        flash("Allowed image types are : png, jpg, jpeg, gif, pdf")
        return(request.url)




@app.route("/display/<filename>")
def display(filename):
    return redirect(url_for('static',filename='uploads/' + filename), code=301)








@app.route('/send_message',methods=['POST'])
def send_message():
    global jfile
    responses={}
    try:
        with open(jfile, "r", encoding='utf-8') as file:
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
                return "I don't know the answer. Please provide a response: "
    

    user_message= request.get_json()["message"]
    if user_message.lower() == "bye":
        
        reply = "Goodbye!"
    else:
        reply= response(user_message)
        if reply.startswith("I don't know the answer. Please provide a response: "):
            # Extract the original user query
            original_query = user_message
            
            # Extract the user's response from the frontend
            user_response = request.get_json()["user_response"]
            while(user_response==""):
                user_response = request.get_json()["user_response"]
            # Save the user's response to the JSON file
            responses[original_query] = user_response
            with open(jfile, "w", encoding='utf-8') as file:
                json.dump(responses, file, ensure_ascii=False, indent=4)   
            
            return jsonify({"message":"Thankyou for providing a response!!"})
    
    return jsonify({"message":reply})
    





if __name__ =="__main__":
    app.run()
