from flask import Flask,render_template,redirect,request
from flask_cors import CORS
import numberDetection.numberDetectionTools
import numpy as np

app = Flask(__name__,template_folder="web")
CORS(app)

BASE = ""

@app.route('/', methods=['GET'])
def redirect_temporarily():
    return redirect(BASE+"numberDetection")

@app.route('/numberDetection', methods=['GET'])
def serve_page():
    return render_template("numberDetection.html")

# Endpoint GET
@app.route('/numberDetection/getData', methods=['GET'])
def post_data():
    data = numberDetection.numberDetectionTools.getTestData()
    return dict({"data":data,"agentAnswer":numberDetection.numberDetectionTools.getNetworkAnswer(np.append([],data["data"])).tolist()}), 200

# Lancer le serveur
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)