import json
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()




def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    #print(sentence_words)
    return sentence_words



def bow(sentence, words,val,id, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    test =[]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w.lower() == s :
                test.append(w)
                if show_details:
                    print("Found Token: %s" % w)
                break
    if len(test)>0:
        val=val+id
    #print(test)
    return val,test


def bow2(sentence, words,val, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    tag=""
    tagmatch=""
    command=""
    command1=""
    command2=""
    command3="" 
    context=""
    bag = [0] * len(words)
    for s in sentence_words:
        for w in words['intents']:
            if w['tag'] == val and s in w['textmatch']:
                tag=w['tag']
                tagmatch=w['textmatch']
                command=w['command']
                command1=w['command1']
                command2=w['command2']
                command3=w['command3']
                context=w['context']
                print("The intent: %s" % s)
                break
                if show_details:
                    print("found in bag: %s" % w)
    return tag,tagmatch,command,command1,command2,command3,context


def classify_text(msg):
    xls = pd.ExcelFile('Dataset_new.xlsx')
    df_item = pd.read_excel(xls, 'Items')
    df_location = pd.read_excel(xls, 'Storage')
    df_vendor = pd.read_excel(xls, 'Vendor')
    df_order = pd.read_excel(xls, 'Order')
    df_po = pd.read_excel(xls, 'PurchaseOrder')
    df_so = pd.read_excel(xls, 'SalesOrder')
    data_file = open("query_intent.json").read()
    textmatch = json.loads(data_file)
    item=df_item['ItemID'].values
    location=df_location['AvailableLocations'].values
    vendor=df_vendor['VendorID'].values
    order=df_order['OrderID'].values
    po=df_po['PONumber'].values
    so=df_so['SONumber'].values
    vendorname=df_vendor['VendorName'].values
    val=""
    text=""
    val,itemlist = bow(msg,item,val,'I',show_details=True)
    val,locationlist = bow(msg,location,val,'L',show_details=True)
    val,vendorlist = bow(msg,vendor,val,'V',show_details=True)
    val,orderlist = bow(msg,order,val,'O',show_details=True)
    val,polist = bow(msg,po,val,'P',show_details=True)
    val,solist = bow(msg,so,val,'S',show_details=True)
    val,vendornamelist = bow(msg,vendorname,val,'N',show_details=True)

    #print(val)
    tag,tagmatch,command,command1,command2,command3,context = bow2(msg,textmatch,val,show_details=False)
    if len(command)>0:
        print("Predicted Data: %s" % context)
        #print(command,command1,command2,itemlist,vendorlist)
        df_list=eval(command)
        #print(df_list)
        if len(command1)>0:
            df_list=eval(command1)
        #print(df_list)
        if len(command2)>0:
            df_list=eval(command2)
        #print(df_list)
        if len(command3)>0:
            df_list=eval(command3)
        #print(df_list)
        if df_list.shape[0] >0:
            df_to_html=df_list.to_html(index=False)
            str="match found"
            #print(df_to_html)
            return str,df_to_html
        else:
            str="match found"
            df_to_html="No values found for provided input."
            return str,df_to_html
        

    elif len(command)==0 and val!="":
        str="match found"
        df_to_html="No values found for provided input. Please refine your search"
        return str,df_to_html
    else:
        str="no match"
        df_to_html=""
        return str,df_to_html
            
