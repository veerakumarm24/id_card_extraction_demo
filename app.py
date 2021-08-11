import logging
import os

import click
import click_logging
import cv2
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)
click_logging.basic_config(logger)
import json

import spacy
import spacy.cli
from facecompare.demos.face_compare import *
from PIL import Image


@click.command()
@click.option('--input', help='Input the image ')
@click.option('--output', help='Output file for end results')
@click_logging.simple_verbosity_option(logger)
@click.option('--verbose', is_flag=True, help='Enables verbose mode.')
def text(input, output, verbose):
    try:

        with open(input) as f:
            im = Image.open(input)
        extension = input.split(".")[-1] in ("jpg", "jpeg", "png")
        click.echo("Image Received!")
        logger.info("File Successfully Saved")

        if not extension:
            logger.info("Image must be jpg or png format!")
            click.echo("Image must be jpg or png format!")
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text2 = pytesseract.image_to_string(im)
        text2 = text2.replace('\n', ' ')
        text2 = text2.replace('\t', ' ')

        # LOAD THE CUSTOM NER MODEL
        nlp2 = spacy.load(os.getcwd() + '\\Custom_NER')
        doc2 = nlp2(text2)
        p = []
        for ent in doc2.ents:
            p.append({ent.label_: ent.text})
        logger.info("Text Extracted")
        with open(output, 'w') as out:
            for item in p:
                out.write(json.dumps(item))
        status = cropFaces(im)
        if status == True:
            logger.info('Image cropped and saved')
        else:
            logger.info('Image cropped and saved')

    except Exception as e:
        click.echo(e)
        logger.info("Exception Occured")


if __name__ == '__main__':
    text()
