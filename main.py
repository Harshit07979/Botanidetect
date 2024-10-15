import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
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
nltk.download('stopwords') 

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

greet_input = ("hi", "Hi","hello","Hello", "whassup", "how are you?")
greet_res = ("Hi", "Hey", "Hey there!","Hello")

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

ww1= np.load("wp1.npy")
ww2= np.load("wp2.npy")
bw1= np.load("bp1.npy")
bw2= np.load("bp2.npy")

wr1= np.load("wr1.npy")
wr2= np.load("wr2.npy")
br1= np.load("br1.npy")
br2= np.load("br2.npy")

wc1= np.load("wc1.npy")
wc2= np.load("wc2.npy")
bc1= np.load("bc1.npy")
bc2= np.load("bc2.npy")

ws1= np.load("ws1.npy")
ws2= np.load("ws2.npy")
bs1= np.load("bs1.npy")
bs2= np.load("bs2.npy")


img_size= 128
batch_size= 16
channels= 3
epochs= 16



def relu(x):
    return np.maximum(0,x)

def softmax(x):
    ex= np.exp(x- np.max(x, axis=-1, keepdims=True))
    return ex/ ex.sum(axis=-1, keepdims=True)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/potato.html')
def home():
    global jfile
    jfile=""
    return render_template("potato.html")

@app.route('/wheat.html')
def home2():
    global jfile1
    jfile1=""
    return render_template("wheat.html")

@app.route('/rice.html')
def home3():
    global jfile2
    jfile2=""
    return render_template("rice.html")

@app.route('/corn.html')
def home4():
    global jfile3
    jfile3=""
    return render_template("corn.html")

@app.route('/sugarcane.html')
def home5():
    global jfile4
    jfile4=""
    return render_template("sugarcane.html")

@app.route('/education.html')
def home6():
    return render_template("education.html")

@app.route('/service.html')
def home7():
    return render_template("service.html")

@app.route('/about.html')
def home8():
    return render_template("about.html")

@app.route('/potato.html', methods=['POST'])
def upload_image():
    path="potato2" #directory path of image set file

    class_names= os.listdir(path)
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
            jfile="responses.json"
        elif(predicted_class_name=="Potato___Late_Blight"):
            jfile="lateres.json"
        elif(predicted_class_name=="Potato___Healthy"):
            jfile="healres.json"
        print(jfile)
        flash("Image Successfully uploaded and displayed below")
        flash(predicted_class_name)
        return render_template("potato.html", filename=filename, predicted_class_name=predicted_class_name,jfile=jfile)
        
    else:
        flash("Allowed image types are : png, jpg, jpeg, gif, pdf")
        return(request.url)




@app.route("/potato.html/display/<filename>")
def display(filename):
    return redirect(url_for('static',filename='uploads/' + filename), code=301)


@app.route('/wheat.html', methods=['POST'])
def upload_image1():
    path="Wheat" #directory path of image set file

    class_names= os.listdir(path)
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
        hidden_layer = np.dot(input_image, ww1) + bw1
        hidden_activation = relu(hidden_layer)
        pred = np.dot(hidden_activation, ww2) + bw2
        pred = softmax(pred)

        # Interpret the predictions
        predicted_class = np.argmax(pred)
        predicted_class_name = class_names[predicted_class]
        print(predicted_class_name)
        global jfile1
        if(predicted_class_name=="Wheat___Brown_Rust"):
            jfile1="wbrown.json"
        elif(predicted_class_name=="Wheat___Yellow_Rust"):
            jfile1="wyellow.json"
        elif(predicted_class_name=="Wheat___Healthy"):
            jfile1="whealthy.json"
        print(jfile1)
        flash("Image Successfully uploaded and displayed below")
        flash(predicted_class_name)
        return render_template("wheat.html", filename=filename, predicted_class_name=predicted_class_name,jfile1=jfile1)
        
    else:
        flash("Allowed image types are : png, jpg, jpeg, gif, pdf")
        return(request.url)




@app.route("/wheat.html/display/<filename>")
def display1(filename):
    return redirect(url_for('static',filename='uploads/' + filename), code=301)

@app.route('/rice.html', methods=['POST'])
def upload_image2():
    path="Rice" #directory path of image set file

    class_names= os.listdir(path)
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
        hidden_layer = np.dot(input_image, wr1) + br1
        hidden_activation = relu(hidden_layer)
        pred = np.dot(hidden_activation, wr2) + br2
        pred = softmax(pred)

        # Interpret the predictions
        predicted_class = np.argmax(pred)
        predicted_class_name = class_names[predicted_class]
        print(predicted_class_name)
        global jfile2
        if(predicted_class_name=="Rice___Brown_Spot"):
            jfile2="rbrown.json"
        elif(predicted_class_name=="Rice__Leaf_Blast"):
            jfile2="rleaf.json"
        elif(predicted_class_name=="Rice__Healthy"):
            jfile2="rhealthy.json"
        elif(predicted_class_name=="Rice__Neck_Blast"):
            jfile2="rneck.json"
        print(jfile2)
        flash("Image Successfully uploaded and displayed below")
        flash(predicted_class_name)
        return render_template("rice.html", filename=filename, predicted_class_name=predicted_class_name,jfile2=jfile2)
        
    else:
        flash("Allowed image types are : png, jpg, jpeg, gif, pdf")
        return(request.url)




@app.route("/rice.html/display/<filename>")
def display2(filename):
    return redirect(url_for('static',filename='uploads/' + filename), code=301)

@app.route('/corn.html', methods=['POST'])
def upload_image3():
    path="corn" #directory path of image set file

    class_names= os.listdir(path)
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
        hidden_layer = np.dot(input_image, wc1) + bc1
        hidden_activation = relu(hidden_layer)
        pred = np.dot(hidden_activation, wc2) + bc2
        pred = softmax(pred)

        # Interpret the predictions
        predicted_class = np.argmax(pred)
        predicted_class_name = class_names[predicted_class]
        print(predicted_class_name)
        global jfile3
        if(predicted_class_name=="Common_Rust"):
            jfile3="crust.json"
        elif(predicted_class_name=="Gray_Leaf_Spot"):
            jfile3="cgray.json"
        elif(predicted_class_name=="Healthy_Corn"):
            jfile3="chealthy.json"
        elif(predicted_class_name=="Northern_Leaf_Blight"):
            jfile3="cnorth.json"
        print(jfile3)
        flash("Image Successfully uploaded and displayed below")
        flash(predicted_class_name)
        return render_template("corn.html", filename=filename, predicted_class_name=predicted_class_name,jfile3=jfile3)
        
    else:
        flash("Allowed image types are : png, jpg, jpeg, gif, pdf")
        return(request.url)




@app.route("/corn.html/display/<filename>")
def display3(filename):
    return redirect(url_for('static',filename='uploads/' + filename), code=301)


@app.route('/sugarcane.html', methods=['POST'])
def upload_image4():
    path="sugarcane" #directory path of image set file

    class_names= os.listdir(path)
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
        hidden_layer = np.dot(input_image, ws1) + bs1
        hidden_activation = relu(hidden_layer)
        pred = np.dot(hidden_activation, ws2) + bs2
        pred = softmax(pred)

        # Interpret the predictions
        predicted_class = np.argmax(pred)
        predicted_class_name = class_names[predicted_class]
        print(predicted_class_name)
        global jfile4
        if(predicted_class_name=="Bacterial_Blight"):
            jfile4="sbacterial.json"
        elif(predicted_class_name=="Red_Rot"):
            jfile4="sred.json"
        elif(predicted_class_name=="Healthy"):
            jfile4="shealthy.json"
        print(jfile4)
        flash("Image Successfully uploaded and displayed below")
        flash(predicted_class_name)
        return render_template("sugarcane.html", filename=filename, predicted_class_name=predicted_class_name,jfile4=jfile4)
        
    else:
        flash("Allowed image types are : png, jpg, jpeg, gif, pdf")
        return(request.url)




@app.route("/sugarcane.html/display/<filename>")
def display4(filename):
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
            msg="Please enter a response... we don't have the answer regarding this";
            cnt=1
            while(user_response==""):
                if(cnt==1):
                    cnt=cnt+1
                    return jsonify({"message":"Please enter a response before continuing..as we don't have this in our data.."})
                user_response = request.get_json()["user_response"]
            # Save the user's response to the JSON file
            responses[original_query] = user_response
            with open(jfile, "w", encoding='utf-8') as file:
                json.dump(responses, file, ensure_ascii=False, indent=4)   
            
            return jsonify({"message":"Thankyou for providing a response!!","msg":msg})
    
    return jsonify({"message":reply})
    

@app.route('/send_message1',methods=['POST'])
def send_message1():
    global jfile1
    responses={}
    try:
        with open(jfile1, "r", encoding='utf-8') as file:
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
            msg="Please enter a response... we don't have the answer regarding this";
            cnt=1
            while(user_response==""):
                if(cnt==1):
                    cnt=cnt+1
                    return jsonify({"message":"Please enter a response before continuing..as we don't have this in our data.."})
                user_response = request.get_json()["user_response"]
            # Save the user's response to the JSON file
            responses[original_query] = user_response
            with open(jfile1, "w", encoding='utf-8') as file:
                json.dump(responses, file, ensure_ascii=False, indent=4)   
            
            return jsonify({"message":"Thankyou for providing a response!!","msg":msg})
    
    return jsonify({"message":reply})
    


@app.route('/send_message2',methods=['POST'])
def send_message2():
    global jfile2
    responses={}
    try:
        with open(jfile2, "r", encoding='utf-8') as file:
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
            msg="Please enter a response... we don't have the answer regarding this";
            cnt=1
            while(user_response==""):
                if(cnt==1):
                    cnt=cnt+1
                    return jsonify({"message":"Please enter a response before continuing..as we don't have this in our data.."})
                user_response = request.get_json()["user_response"]
            # Save the user's response to the JSON file
            responses[original_query] = user_response
            with open(jfile2, "w", encoding='utf-8') as file:
                json.dump(responses, file, ensure_ascii=False, indent=4)   
            
            return jsonify({"message":"Thankyou for providing a response!!","msg":msg})
    
    return jsonify({"message":reply})
    
@app.route('/send_message3',methods=['POST'])
def send_message3():
    global jfile3
    responses={}
    try:
        with open(jfile3, "r", encoding='utf-8') as file:
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
            msg="Please enter a response... we don't have the answer regarding this";
            cnt=1
            while(user_response==""):
                if(cnt==1):
                    cnt=cnt+1
                    return jsonify({"message":"Please enter a response before continuing..as we don't have this in our data.."})
                user_response = request.get_json()["user_response"]
            # Save the user's response to the JSON file
            responses[original_query] = user_response
            with open(jfile3, "w", encoding='utf-8') as file:
                json.dump(responses, file, ensure_ascii=False, indent=4)   
            
            return jsonify({"message":"Thankyou for providing a response!!","msg":msg})
    
    return jsonify({"message":reply})
    
@app.route('/send_message4',methods=['POST'])
def send_message4():
    global jfile4
    responses={}
    try:
        with open(jfile4, "r", encoding='utf-8') as file:
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
            msg="Please enter a response... we don't have the answer regarding this";
            cnt=1
            while(user_response==""):
                if(cnt==1):
                    cnt=cnt+1
                    return jsonify({"message":"Please enter a response before continuing..as we don't have this in our data.."})
                user_response = request.get_json()["user_response"]
            # Save the user's response to the JSON file
            responses[original_query] = user_response
            with open(jfile4, "w", encoding='utf-8') as file:
                json.dump(responses, file, ensure_ascii=False, indent=4)   
            
            return jsonify({"message":"Thankyou for providing a response!!","msg":msg})
    
    return jsonify({"message":reply})
    



if __name__ =="__main__":
    app.run()
