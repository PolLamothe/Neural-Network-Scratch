from flask import Flask,render_template,redirect,request
from flask_cors import CORS
import sys
sys.path.append("../")
import numberDetection.numberDetectionTools
import numpy as np
import snake.trainSnakeEvoTools

app = Flask(__name__,template_folder="")
CORS(app)

BASE = "/IA/"

@app.route('/', methods=['GET'])
def redirect_temporarily():
    return render_template("index.html")

@app.route('/numberDetection', methods=['GET'])
def serve_numberDetection_page():
    return render_template("numberDetection.html")

# Endpoint GET
@app.route('/numberDetection/getData', methods=['GET'])
def post_numberDetection_data():
    data = numberDetection.numberDetectionTools.getTestData()
    return dict({"data":data,"agentAnswer":numberDetection.numberDetectionTools.getNetworkAnswer(np.append([],data["data"])).tolist()}), 200

@app.route('/snake', methods=['GET'])
def serve_snake_page():
    return render_template("snake.html")

@app.route('/snake/getData', methods=['GET'])
def post_snake_data():
    data = snake.trainSnakeEvoTools.getWholeGameData()
    return data, 200

# Lancer le serveur
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)