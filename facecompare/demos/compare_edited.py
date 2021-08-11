#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import itertools
import os
import xlwt 
from xlwt import Workbook

import numpy as np
np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
print(fileDir)
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
face_comparision_threshold = 0.9

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
print(args.verbose)
if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
print(align)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
print("net")
modelName = args.networkModel.rsplit('/')[4]

print("Argument image dimension")
print(args.imgDim)
print(args.imgs)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    print(imgPath)
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    cv2.rectangle(bgrImg, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (0,255,255), 2)
    # cv2.imshow('detected',bgrImg)
    # cv2.waitKey(0)
    # cv2.imwrite('detected/' + image1, face)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    # print("representation")
    # print(rep)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep

# for (img1, img2) in itertools.combinations(args.imgs, 2):
#     print("Argument Image")
#     print(img1)
#     print(img2)
    
#     # print(getRep(img1))
#     # print(getRep(img2))
    
#     d = getRep(img1) - getRep(img2)
#     print("Comparing {} with {}.".format(img1, img2))
#     print("  + Squared l2 distance between representations: {:0.3f}".format(np.dot(d, d)))
wb = Workbook() 
# wb1 = Workbook() 

sheet1 = wb.add_sheet(modelName,cell_overwrite_ok=True)
# sheet2 = wb1.add_sheet(modelName,cell_overwrite_ok=True)
sheet1.write(0, 0, "S No")
sheet1.write(0, 1, "Source Image")
sheet1.write(0, 2, "Destination Image")
sheet1.write(0, 3, "Score")
# sheet2.write(0, 0, "S No")
# sheet2.write(0, 1, "Source Image")
# sheet2.write(0, 2, "Destination Image")
# sheet2.write(0, 3, "Score") 
for x in args.imgs:
    inputFolder = x

folderName = str(inputFolder).split("/")[3]
print("input folder" , inputFolder)
rowindex = 1
for folder in os.listdir(inputFolder):
    # print("folder")
    # print(folder)
    
    subFolder = inputFolder+"/"+folder
    faces = []
    for image in os.listdir(subFolder):
        # print(image)
        faces.append(image)
    # for (img1, img2) in itertools.combinations(faces, 2):
    #     print("Argument Image")
    #     print(img1)
    #     print(img2)
        
        # print(getRep(img1))
        # print(getRep(img2))
    sheet1.write(rowindex, 0, rowindex)
    sheet1.write(rowindex, 1, faces[1])
    sheet1.write(rowindex, 2, faces[0])
       
    d = getRep(subFolder+"/"+faces[1]) - getRep(subFolder+"/"+faces[0])
    print("Comparing {} with {}.".format(subFolder+"/"+faces[1], subFolder+"/"+faces[0]))
    score = format(np.dot(d, d))
    # print(type(score))
    # print(type(face_comparision_threshold))
    if float(score) <= face_comparision_threshold:
        sheet1.write(rowindex,4,"Matched")
    else:
        sheet1.write(rowindex,4,"Not Matched")
    print(" ########################################################## Squared l2 distance between representations: {:0.3f}", score)
    sheet1.write(rowindex, 3, score)
    rowindex = rowindex + 1

filename = folderName+"_"+modelName
wb.save('excel/'+folderName+'/'+filename+'.xls')

    
#     # print(getRep(img1))
#     # print(getRep(img2))
    
#     d = getRep(img1) - getRep(img2)
#     print("Comparing {} with {}.".format(img1, img2))
#     print("  + Squared l2 distance between representations: {:0.3f}".format(np.dot(d, d)))
