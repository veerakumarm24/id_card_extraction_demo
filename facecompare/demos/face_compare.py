import time
start = time.time()
import argparse
import cv2
import itertools
import os
import numpy as np
import sys
# sys.path.insert(0, '../../../openface-master')
import facecompare.openface as openface
import pickle

np.set_printoptions(precision=2)
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
parser = argparse.ArgumentParser()
dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
networkModel = os.path.join(openfaceModelDir, 'torch_models/nn4.small1.v1.t7')
print(networkModel)
imgDim = 96
verbose = False
rgbImg = 0

if verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, imgDim)

if verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))

#Compare faces
def facecompare(selfie,proof):
    start = time.time()
    print("Face compare initiated")
    selfie_rep = getRep(selfie,1)
    status = selfie_rep['status']
    if status != 200:
        return selfie_rep

    proof_rep = getRep(proof,2)
    status = proof_rep['status']
    if status != 200:
        return proof_rep
    
    score = faceDistanceScore(selfie_rep['rep'],proof_rep['rep'])
    print("Face Score " , score)
    print("Facecompare total time took {} seconds.".format(
        time.time() - start))
    return {"status":200,"score" : score, "s_rep":selfie_rep['rep'],
    "s_image" :selfie_rep['image'],"p_image":proof_rep['image'],"p_c_image":proof_rep['c_image']}

def getFaceRep(selfie):
    selfie_rep = getRep(selfie,1)
    status = selfie_rep['status']
    return selfie_rep

def faceDistanceScore(source_rep , test_rep):
    d = source_rep - test_rep
    score = format(np.dot(d, d))
    return score


#Similarity between the two face
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def getFaceBoundingBoxes(bgrImg):
    global rgbImg
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    
    if verbose:
        print("  + Original size: {}".format(rgbImg.shape))
        
    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    return bb

#Get face representation
def getRep(img, fromwhere):
    processtime = time.time()
    # print("Rep Thread process started ",fromwhere)
    response = {}
    response['status'] = 200
    if verbose:
        print("Processing {}.".format(img))
    # convert string data to numpy array
    npimg = np.fromstring(img, np.uint8)
    # convert numpy array to image
    image = cv2.imdecode(npimg, 1)
    bgrImg = image #cv2.imread(imgPath) 
    if bgrImg is None:
        response['status'] = 500
        response['message'] = "Unable to load image"
        return response

    i = 0
    while i < 4:
        if i == 0:
            bb = getFaceBoundingBoxes(bgrImg)
            if bb is not None:
                break
        if i == 1:
            bgrImg = cv2.rotate(bgrImg, cv2.ROTATE_90_CLOCKWISE)
            bb = getFaceBoundingBoxes(bgrImg)
            if bb is not None:
                break
        if i == 2:
            bgrImg = cv2.rotate(bgrImg, cv2.ROTATE_90_CLOCKWISE)
            bb = getFaceBoundingBoxes(bgrImg)
            if bb is not None:
                break
        if i == 3:
            bgrImg = cv2.rotate(bgrImg, cv2.ROTATE_90_CLOCKWISE)
            bb = getFaceBoundingBoxes(bgrImg)
            if bb is not None:
                break
        i = i + 1

    if bb is None:
        response['noface'] = fromwhere
        response['status'] = 500
        response['message'] = "Unable to find a face,kindly retake again"
        return response

    response['image'] = bgrImg

    cv2.rectangle(bgrImg, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (0,255,255), 2)

    crop_img = bgrImg[ bb.top():bb.bottom(), bb.left():bb.right()]

    # cv2.imshow("image", crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        response['status'] = 500
        response['message'] = "Unable to align image"
        return response
    if verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    # print("rep started")
    # print("SHAPE:", alignedFace.shape)
    rep = net.forward(alignedFace)
    # print("rep",rep)
    if verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    response['rep'] = rep
    response['c_image'] = crop_img
    # print(fromwhere, " Thread Process total time took {} seconds.".format(time.time() - processtime))
    return response




def cropFaces(image):
    print(type(image))
    selfie_rep = getRep(image,1)
    if(selfie_rep['status']==200):
        return True
    else:
        return False