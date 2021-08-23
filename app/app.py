# -*- coding: utf-8 -*-

from flasgger import Swagger
from flask import Flask, request, jsonify

import predict

swagger_config = Swagger.DEFAULT_CONFIG
swagger_config['swagger_ui_bundle_js'] = '//unpkg.com/swagger-ui-dist@3.3.0/swagger-ui-bundle.js'
swagger_config['swagger_ui_standalone_preset_js'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui-standalone-preset.js'
swagger_config['jquery_js'] = '//unpkg.com/jquery@2.2.4/dist/jquery.min.js'
swagger_config['swagger_ui_css'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui.css'

app = Flask(__name__)


def create_swagger(app_):
    template = {
        "openapi": '3.0.0',
        "info": {
            "title": "Automatic Image Assessment API by NiNe Capture",
            "version": "3.0",
            "description": "Here at NiNe Capture, we are passionate about pioneering cutting-edge technologies to maximize both the creativity and productivity of our services. In this API, we leverage deep convolutional neural networks to automatically evaluate and prioritize client photos. Having been trained on millions of high-quality open-source photographs and fine-tuned with thousands our own best works, our models are capable of processing and evaluating both the technical and aesthetic merit of thousands of photographs in a matter of seconds. This ensures that our clients receive unparalleled, high-quality work in record time."
        },
        "basePath": "/",
        "schemes": [
            "http",
            "https"
        ]
    }
    return Swagger(app_, template=template)


swag = create_swagger(app)


@app.route('/')
def index():
    from flask import request
    return f"Please refer to the <a href=\"{request.url_root}apidocs\">API documentation</a> for more details."


@swag.definition('Prediction')
class Prediction(object):
    """
    Prediction Object
    ---
    type: object
    properties:
      base_model_name:
        type: string
        example: MobileNet
      weights_aesthetic:
        type: string
        example: weights_aesthetic.hdf5
      weights_technical:
        type: string
        example: weights_technical.hdf5
      image_source:
        type: string
        example: "test_images/"
    """
    def __init__(self, text, ratio):
        self.text = str(text)
        self.ratio = str(ratio)

    def dump(self):
        return dict(vars(self).items())


@app.route('/api/predict', methods=["GET", "POST"])
def image_assessment():
    """Click here to test out sample data.
    Press the `Try it out` button below to test out sample data against the prediction REST API.
    ---
    tags:
      - name: Image Assessment
    parameters:
      - name: base_model_name
        in: query
        type: string
        required: true
        description: Base model specification
        default: 'MobileNet'

      - name: weights_aesthetic
        in: query
        type: string
        required: true
        description: Weights file for the aesthetic model
        default: 'weights_aesthetic.hdf5'

      - name: weights_technical
        in: query
        type: string
        required: true
        description: Weights file for the technical model
        default: 'weights_technical.hdf5'

      - name: image_source
        in: query
        type: string
        required: true
        description: Path to an image file or file directory containing images
        default: 'test_images/'
    """
    # Return GET output if called directly from the REST API.
    if request.method == 'GET':
        base_model_name = request.args.get("base_model_name")
        weights_aesthetic = request.args.get("weights_aesthetic")
        weights_technical = request.args.get("weights_technical")
        image_source = request.args.get("image_source")
        prediction_dict = predict.main(
            base_model_name=base_model_name,
            weights_aesthetic=weights_aesthetic,
            weights_technical=weights_technical,
            image_source=image_source)
        print(prediction_dict)
        return jsonify(prediction_dict)
    elif request.method == 'POST':
        base_model_name = request.json.get("base_model_name")
        weights_aesthetic = request.json.get("weights_aesthetic")
        weights_technical = request.json.get("weights_technical")
        image_source = request.json.get("image_source")
        prediction_dict = predict.main(
            base_model_name=base_model_name,
            weights_aesthetic=weights_aesthetic,
            weights_technical=weights_technical,
            image_source=image_source)
        return jsonify(prediction_dict)


@app.route('/api/rename', methods=["POST"])
def rename():
    base_model_name = request.args.get("base_model_name")
    weights_aesthetic = request.args.get("weights_aesthetic")
    weights_technical = request.args.get("weights_technical")
    image_source = request.args.get("image_source")
    base_model_name = base_model_name,
    weights_aesthetic = weights_aesthetic,
    weights_technical = weights_technical,
    image_source = image_source
    prediction_dict = predict.rename_files(
        base_model_name=base_model_name,
        weights_aesthetic=weights_aesthetic,
        weights_technical=weights_technical,
        image_source=image_source)
    return jsonify(prediction_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
