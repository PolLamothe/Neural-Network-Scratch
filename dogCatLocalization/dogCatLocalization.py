import argparse
import copy
import os
import pickle
import random
import sys
import time
import numpy as np
from PIL import Image
import scipy.ndimage
sys.path.append("../")
import classe
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

PATH_TO_XML = "/annotations/xmls/"

PATH_TO_IMG = "/images/"

def imageToMatrix(path : str) -> np.ndarray:
    image = Image.open(path).convert("RGB")

    matrix = np.array(image)
    resized_matrix = np.stack([scipy.ndimage.zoom(matrix[:, :, i], (128/matrix.shape[0],128/matrix.shape[1])) for i in range(3)], axis=-1)
    resized_matrix =  np.transpose(resized_matrix, (2, 0, 1))

    return resized_matrix/255

def checkAnswer(right, response):
    greater = None
    for i in range(2):
        if(greater == None):greater = (i,response[i])
        else:
            if(response[i] > greater[1]):greater = (i,response[i])
    if(greater[0] == right):return True
    else:return False

def getTrainedNetwork() -> classe.Network:
    with open(os.path.dirname(os.path.realpath(__file__))+"/dogCatLocalization.pkl", "rb") as file:
        return pickle.load(file)
    
TRAINING_SIZE = 3000
if(TRAINING_SIZE > 3686):
    raise Exception("Training size too high")

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", dest="loadTrainedModel", action='store_true')
    parser.add_argument("-c", dest="help", action='store_true')
    parser.add_argument("-s", dest="save", action='store_true')
    args = parser.parse_args()
    useTrainedModel = args.loadTrainedModel
    if(args.help):
        print("-c : Get all command")
        print("-l : Use the trained model")
        print("-s : Save the model once it has finish training")
        exit(0)
    
    if(not args.loadTrainedModel):

        print("starting training data importation")

        directory = Path("./assets/"+PATH_TO_XML)

        files = [f.name for f in directory.iterdir() if f.is_file()]

        files = files[:TRAINING_SIZE]

        datas = []

        for file in files:
            tree = ET.parse("./assets"+PATH_TO_XML+file)
            root = tree.getroot()

            obj = root.findall("object")[0]

            datas.append({
                "filename" : root.find("filename").text,
                "name": obj.find("name").text,
                "bndbox": {
                    "xmin": int(obj.find("bndbox/xmin").text)/int(root.find("size/width").text),
                    "ymin": int(obj.find("bndbox/ymin").text)/int(root.find("size/height").text),
                    "xmax": int(obj.find("bndbox/xmax").text)/int(root.find("size/width").text),
                    "ymax": int(obj.find("bndbox/ymax").text)/int(root.find("size/height").text),
                }
            })

        images = []
        for i in range(TRAINING_SIZE):
            images.append(imageToMatrix("./assets"+PATH_TO_IMG+datas[i]["filename"]))
        
        print("training data imported")
        
        KERNELS_SIZE = [3,3,3,3,3]

        KERNELS_NUMBER = [8,16,32,64,128]

        LEARNING_RATE = 0.01

        BATCH_SIZE = 32

        network1 = classe.CNN([
            classe.PoolingLayer(32,16,depth=KERNELS_NUMBER[2],batch_size=BATCH_SIZE),
            classe.ConvolutionalLayer(16,16,KERNELS_SIZE[3],KERNELS_NUMBER[3],depth=KERNELS_NUMBER[2],learning_rate=LEARNING_RATE,activation=classe.Relu,batch_size=BATCH_SIZE),
            classe.BatchNormalization(learning_rate=LEARNING_RATE),
            classe.PoolingLayer(16,8,depth=KERNELS_NUMBER[3],batch_size=BATCH_SIZE),
            classe.ConvolutionalLayer(8,8,KERNELS_SIZE[4],KERNELS_NUMBER[4],depth=KERNELS_NUMBER[3],learning_rate=LEARNING_RATE,activation=classe.Relu,batch_size=BATCH_SIZE),
            classe.BatchNormalization(learning_rate=LEARNING_RATE),
            classe.FlateningLayer(8,KERNELS_NUMBER[-1],batch_size=BATCH_SIZE),
            classe.FullyConnectedLayer(8**2*KERNELS_NUMBER[-1],512,classe.Tanh,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
            classe.Dropout(0.2),
            classe.FullyConnectedLayer(512,2,classe.Sigmoid,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
        ],BATCH_SIZE)

        network2 = classe.CNN([
            classe.FlateningLayer(32,KERNELS_NUMBER[2],batch_size=BATCH_SIZE),
            classe.FullyConnectedLayer(32**2*KERNELS_NUMBER[2],512,classe.Tanh,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
            classe.FullyConnectedLayer(512,4,classe.Sigmoid,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
        ],BATCH_SIZE)

        network = classe.Network([
            classe.ConvolutionalLayer(128,128,KERNELS_SIZE[0],KERNELS_NUMBER[0],depth=3,learning_rate=LEARNING_RATE,activation=classe.Relu,batch_size=BATCH_SIZE),
            classe.BatchNormalization(learning_rate=LEARNING_RATE),
            classe.PoolingLayer(128,64,depth=KERNELS_NUMBER[0],batch_size=BATCH_SIZE),
            classe.ConvolutionalLayer(64,64,KERNELS_SIZE[1],KERNELS_NUMBER[1],depth=KERNELS_NUMBER[0],learning_rate=LEARNING_RATE,activation=classe.Relu,batch_size=BATCH_SIZE),
            classe.BatchNormalization(learning_rate=LEARNING_RATE),
            classe.PoolingLayer(64,32,depth=KERNELS_NUMBER[1],batch_size=BATCH_SIZE),
            classe.ConvolutionalLayer(32,32,KERNELS_SIZE[2],KERNELS_NUMBER[2],depth=KERNELS_NUMBER[1],learning_rate=LEARNING_RATE,activation=classe.Relu,batch_size=BATCH_SIZE),
            classe.BatchNormalization(learning_rate=LEARNING_RATE),
            classe.SplitLayer(network1,0.7,network2,0.3),
        ],BATCH_SIZE)

        count = 0

        currentIndex = 0

        COEFF_SIZE = 300

        previousAnswers = []

        startTime = time.time()

        while(count * BATCH_SIZE < TRAINING_SIZE):
            count += 1

            currentIndexs = [int((currentIndex+i)%TRAINING_SIZE) for i in range(BATCH_SIZE)]
            rightsClass = [0 if datas[index]["name"] == "cat" else 1 for index in currentIndexs]
            rightsPos = [datas[index]["bndbox"] for index in currentIndexs]

            currentIndex = (currentIndex + BATCH_SIZE)%TRAINING_SIZE

            result = network.forward(np.array([images[index] for index in currentIndexs]))

            classerror = []
            poserror = []
            mse_class = 0
            mse_pos = 0

            for i in range(len(result[0])):
                rightClass = rightsClass[i]
                rightPos = rightsPos[i]

                if(len(previousAnswers) >= COEFF_SIZE):
                    previousAnswers.pop(0)
                previousAnswers.append(checkAnswer(rightClass,result[0][i]))

                err = []
                for j in range(2):
                    if(j == rightClass):
                        err.append(1-result[0][i][j])
                        mse_class += ((1-result[0][i][j])**2)/2/BATCH_SIZE
                    else:
                        err.append(-result[0][i][j])
                        mse_class += ((-result[0][i][j])**2)/2/BATCH_SIZE

                classerror.append(copy.deepcopy(err))

                mse_pos += (rightPos["xmin"]-result[1][i][0])**2/4/BATCH_SIZE
                mse_pos += (rightPos["ymin"]-result[1][i][1])**2/4/BATCH_SIZE
                mse_pos += (rightPos["xmax"]-result[1][i][2])**2/4/BATCH_SIZE
                mse_pos += (rightPos["ymax"]-result[1][i][3])**2/4/BATCH_SIZE

                poserror.append([rightPos["xmin"]-result[1][i][0],rightPos["ymin"]-result[1][i][1],rightPos["xmax"]-result[1][i][2],rightPos["ymax"]-result[1][i][3]])

            network.backward([np.array(classerror),np.array(poserror)])

            print("nombre d'iterations : "+str(count*BATCH_SIZE)," class mse : ",round(mse_class,2)," class success rate : ",round(previousAnswers.count(True)/len(previousAnswers),2)," pos mse : ",round(mse_pos,2))
            sys.stdout.flush()
        
        network.turnOffTraining()

        if(args.save):
            print("saving your model")
            file_name = 'dogCatLocalization.pkl'
            with open(file_name, 'wb') as file:
                pickle.dump(network, file)
    else:
        TRAINING_SIZE = 2000
        if(TRAINING_SIZE > 3686):
            raise Exception("Training size too high")

        print("starting training data importation")

        directory = Path("./assets/"+PATH_TO_XML)

        files = [f.name for f in directory.iterdir() if f.is_file()]

        files = files[:TRAINING_SIZE]

        datas = []

        for file in files:
            tree = ET.parse("./assets"+PATH_TO_XML+file)
            root = tree.getroot()

            obj = root.findall("object")[0]

            datas.append({
                "filename" : root.find("filename").text,
                "name": obj.find("name").text,
                "bndbox": {
                    "xmin": int(obj.find("bndbox/xmin").text)/int(root.find("size/width").text),
                    "ymin": int(obj.find("bndbox/ymin").text)/int(root.find("size/height").text),
                    "xmax": int(obj.find("bndbox/xmax").text)/int(root.find("size/width").text),
                    "ymax": int(obj.find("bndbox/ymax").text)/int(root.find("size/height").text),
                }
            })

        images = []
        for i in range(TRAINING_SIZE):
            images.append(imageToMatrix("./assets"+PATH_TO_IMG+datas[i]["filename"]))
        
        print("training data imported")

        network = getTrainedNetwork()

        while(True):
            index = random.randint(0,TRAINING_SIZE)

            result = network.forward(np.array([images[index]]))
            print(result)
            image = np.transpose(images[index], (1, 2, 0))

            plt.imshow(image)

            x1, y1 = result[1][0][0]*128, result[1][0][1]*128
            x2, y2 = result[1][0][2]*128, result[1][0][3]*128

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=2, fill=False)
            plt.gca().add_patch(rect)

            plt.axis("off")
            plt.show()

    print("starting testing data importation")

    directory = Path("./assets/"+PATH_TO_XML)

    files = [f.name for f in directory.iterdir() if f.is_file()]

    files = files[TRAINING_SIZE:]

    datas = []

    for file in files:
        tree = ET.parse("./assets"+PATH_TO_XML+file)
        root = tree.getroot()

        obj = root.findall("object")[0]

        datas.append({
            "filename" : root.find("filename").text,
            "name": obj.find("name").text,
            "bndbox": {
                "xmin": int(obj.find("bndbox/xmin").text)/int(root.find("size/width").text),
                "ymin": int(obj.find("bndbox/ymin").text)/int(root.find("size/height").text),
                "xmax": int(obj.find("bndbox/xmax").text)/int(root.find("size/width").text),
                "ymax": int(obj.find("bndbox/ymax").text)/int(root.find("size/height").text),
            }
        })

    images = []
    for i in range(len(datas)):
        images.append(imageToMatrix("./assets"+PATH_TO_IMG+datas[i]["filename"]))

    print("testing data imported")

    successCout = 0
    testCount = 0

    network = getTrainedNetwork()

    for i in range(len(images)):
        result = network.forward(np.array([images[i]]))
        right = 0 
        if datas[i]["name"] != "cat":
            right = 1
        if(checkAnswer(right,result[0][0])):
            successCout += 1
        testCount += 1

    print("The success rate of the model on the testing dataset is : ",round(successCout/testCount,2))