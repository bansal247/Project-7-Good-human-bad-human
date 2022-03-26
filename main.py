from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from predict import good_bad
import base64

# install tensorflow cpu not keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class GbApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = good_bad(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():

    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    clApp = GbApp()
    image = request.json['image']
    imgdata = base64.b64decode(image)
    with open(clApp.filename, 'wb') as f:
        f.write(imgdata)
        f.close()
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', debug=True)
