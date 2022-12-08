import random
import os
import numpy as np
import pickle
import json
import pandas as pd
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from flask import Flask, render_template, request, make_response, send_from_directory
from load_inventory_data import *
import pdfkit
from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = 'C:\Radhika\Coventry\Major project\Chatbot\Message\Files'
    return send_from_directory(directory=uploads, path=filename)

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    print("The User Input: %s" % msg)
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
        return res
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
        return res
    elif msg.startswith('POCreation:'):
        if ',' in msg:
            res = po_creation(msg)
            return res
        else:
            res= "Please provide itemid and quantity. Ex POCreation: ItemID11,50"
            return res
    else:
        str,res = classify_text(msg)
        if str == "match found":
            return res
        else:
            ints = predict_class(msg, model)
            res = getResponse(ints, intents)
            return res
    
def po_creation(msg):
    try:
        xls = pd.ExcelFile('Dataset_new.xlsx')
        df1 = pd.read_excel(xls, 'Items')
        df2 = pd.read_excel(xls, 'Storage')
        df_vendor = pd.read_excel(xls, 'Vendor')
        #name="POCreation: ItemID11,50"
        name=msg.replace("POCreation: ","")
        Item=name.split(",")[0]
        Quantity=name.split(",")[1]
        Quantity=int(Quantity)
        print(Item)
        print(Quantity)
        #print(df1)
        df_vendor=df_vendor.loc[df_vendor['ItemID'] == Item]
        df_vendor.loc[df_vendor['VendorHeldQuantity'] >= Quantity, 'Quantity_Available'] = 'Yes'
        df_vendor.loc[df_vendor['VendorHeldQuantity'] < Quantity, 'Quantity_Available'] = 'No'
        df_vendor=df_vendor.loc[df_vendor['Quantity_Available'] == 'Yes']
        df_vendor=df_vendor.loc[df_vendor['ItemPrice'] == df_vendor['ItemPrice'].min()]
        df_vendor['Quantity']=Quantity
        if df_vendor.shape[0] >0:
            df_to_html=df_vendor.to_html(index=False)
            date=datetime.today().strftime('%Y%m%d%H%M%S')
            html_file = 'C:\Radhika\Coventry\Major project\Chatbot\Message\Files\PO_Creation_'+date+'.html'
            pdf_file = 'C:\Radhika\Coventry\Major project\Chatbot\Message\Files\PO'+date+'.pdf'
            pdf_name = 'PO'+date+'.pdf'
            path_wkhtmltopdf = r'C:\Radhika\Coventry\Major project\Chatbot\Message\wkhtmltox\bin\wkhtmltopdf.exe'
            config=pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
            env=Environment(loader=FileSystemLoader("templates/"))
            template = env.get_template('PO_Creation.html')
            template_vars = {"PO_Creation": df_to_html,"Purchase_Order":'PO'+date} 
            html_out = template.render(template_vars)
            with open(html_file, "w") as file:
                file.write(html_out)
            pdfkit.from_file(html_file,pdf_file,configuration=config)
            result='<h2>Purchase Order Created: PO'+date+'</h2><a href="http://127.0.0.1:5000/uploads/'+pdf_name+'" download="'+pdf_name+'">PO'+date+'</a>'
            return(result)
        else:
            result="Unable to create PO with details provided. Please try with appropriate values"
            return(result)
    except:
        result="Unable to create PO with details provided. Please try with appropriate values"
        return(result)



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words



def bot_search(sentence, words, show_details=True):
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
    p = bot_search(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


if __name__ == "__main__":
    app.run()

