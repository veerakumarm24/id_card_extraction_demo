# id_card_extraction_demo

In this project work, we have established an algorithm through the virtue of which we can detect necessary textual parameters of the ID Card and the candidate’s image from the ID Card.  So, the below steps need to be done to test the model’s performance on an ID Card.

First, we will open a terminal and will take a pull by this command:

## git clone https://github.com/veerakumarm24/id_card_extraction_demo.git

Next, we will have to install the requirements.txt in our local by running the below command.

## pip install -r requirements.txt

Next, we will install the below libraries similarly one after another:

## pip install nltk

 

## pip install spacy==2.3.5

 

## pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

 

## pip install pyresparser

 

For our POC work we are working with Pytesseract, so we need to install the the library and its dependencies. To install the library please execute the below command:

## pip install pytesseract

 

Next, we will download click library by the following commands

 

## pip install click

 

As a final step we can just run the below command using one ID card to extract the relevant information from the card.
 

## python app.py --input ID5.jpg --output IDF.txt –verbose

 

If you look at our repository then you will find the ID5.jpg and IDF.txt, you can get an idea about the model performance.

 

Note: As we have trained our NER model on a smaller dataset, so for now the model is able to extract information from some images which has slightly better quality and it can be able to track the pattern. So, we have given a list of images inside the ID Card Images folder that you can utilize for your testing purpose.