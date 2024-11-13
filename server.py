from flask import Flask
from flask_cors import CORS
import numberDetection.numberDetectionTools
import numpy as np

app = Flask(__name__)
CORS(app)

# Endpoint GET
@app.route('/numberDetection/getData', methods=['GET'])
def post_data():
    data = numberDetection.numberDetectionTools.getTestData()
    return dict({"data":data,"agentAnswer":numberDetection.numberDetectionTools.getNetworkAnswer(np.append([],data["data"])).tolist()}), 200

# Lancer le serveur
if __name__ == '__main__':
    app.run(port=5000)