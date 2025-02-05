from flask import Flask,render_template,redirect,request
from flask_cors import CORS
import sys
sys.path.append("../")
import numberDetection.numberDetectionTools
import imageDetection.numberDetectionTools
import numpy as np
import snake.trainSnakeEvoTools

app = Flask(__name__,template_folder="")
CORS(app)

BASE = "/IA/"

@app.route('/', methods=['GET'])
def redirect_temporarily():
    return render_template("index.html")

@app.route('/numberDetection/<method>', methods=['GET'])
def serve_numberDetection_page(method):
    if(method == "simple"):
        return render_template("numberDetectionSimple.html")
    elif(method == "convolution"):
        return render_template("numberDetectionConvolution.html")

# Endpoint GET
@app.route('/numberDetection/<method>/getData', methods=['GET'])
def post_numberDetection_data(method):
    if(method == "simple"):
        data = numberDetection.numberDetectionTools.getTestData()
        return dict({"data":data,"agentAnswer":numberDetection.numberDetectionTools.getNetworkAnswer(np.append([],data["data"])).tolist()}), 200
    elif(method == "convolution"):
        data = imageDetection.numberDetectionTools.getTestData()
        answer = imageDetection.numberDetectionTools.getNetworkAnswer(np.array([data["data"]]))
        return dict({
            "data":data,
            "agentAnswer":answer["answer"].tolist(),
            "convolution":answer["convolution"].tolist()
            }), 200

@app.route('/snake', methods=['GET'])
def serve_snake_page():
    return render_template("snake.html")

@app.route('/snake/getData', methods=['GET'])
def post_snake_data():
    data = snake.trainSnakeEvoTools.getWholeGameData()
    return data, 200

# Lancer le serveur
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)