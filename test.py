# import uvicorn
# from fastapi import FastAPI,File, UploadFile
from flask import Flask,request

import pytesseract
from pytesseract import Output
import cv2
from PIL import Image
import spacy
import random
from io import BytesIO
import spacy.cli
from facecompare.demos.face_compare import *

app = Flask(__name__)
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


@app.route("/idcardtext",methods=['POST'])
def text():
    file=request.files['file1']
    im = Image.open(file)
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    print(extension)
    #im = cv2.bilateralFilter((im),9,75,75)
    #im = cv2.GaussianBlur(im, (5, 5), 0)
    if not extension:
        return "Image must be jpg or png format!"
    text2 = pytesseract.image_to_string(im)
    text2 = text2.replace('\n',' ')
    text2 = text2.replace('\t',' ')
    #LOAD THE CUSTOM NER MODEL
    nlp2 = spacy.load("/home/desktop-pr-27/Desktop/ID_Card_Extractor/Custom_NER")
    doc2 = nlp2(text2)
    p=[]
    for ent in doc2.ents:
        p.append({ent.label_:ent.text})
        #print(str(ent.label_)+": " +str(ent.text))
    return {
        'Extracted Output': p
    }

 
@app.post("/idcardface")
def face():
    image = request.files['file1'].read()
    status=cropFaces(image)
    if status==True:
        return {'status':200,'message':'Image cropped and saved'}
    else:
        return {'status':500,'message':'Unable to save image'}


if __name__ == '__main__':
    # uvicorn.run(app, host='127.0.0.1', port=8000)
    app.run(debug=True)
#uvicorn app:app --reload