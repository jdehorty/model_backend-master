# -*- coding: utf-8 -*-

from flasgger import Swagger
from flask import Flask, request, jsonify
from joblib import load

import summarizer

swagger_config = Swagger.DEFAULT_CONFIG
swagger_config['swagger_ui_bundle_js'] = '//unpkg.com/swagger-ui-dist@3.3.0/swagger-ui-bundle.js'
swagger_config['swagger_ui_standalone_preset_js'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui-standalone-preset.js'
swagger_config['jquery_js'] = '//unpkg.com/jquery@2.2.4/dist/jquery.min.js'
swagger_config['swagger_ui_css'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui.css'

app = Flask(__name__)


def create_swagger(app_):
    template = {
        "openapi": '3.0.1',
        "info": {
            "title": "Swagger API - Copilot Text Summarization",
            "version": "3.0",
            "description": "OpenAPI 3.0 Specification for the REST-API backend for text summarizer",
        },
        "basePath": "/",
        "schemes": [
            "http",
            "https"
        ]
    }
    return Swagger(app_, template=template)


swag = create_swagger(app)


@swag.definition('Text')
class Text(object):
    """
    Text Object
    ---
    type: object
    properties:
        text:
            type: string
        ratio:
            type: string
    """
    def __init__(self, text, ratio):
        self.text = str(text)
        self.ratio = str(ratio)

    def dump(self):
        return dict(vars(self).items())


@app.route('/')
def index():
    from flask import request
    return f"Please refer to the <a href=\"{request.url_root}apidocs\">API documentation</a> for more details."


@app.route('/api/summarize', methods=["GET", "POST"])
def summarize_text():
    """Click here to test out sample data.
    Press the `Try it out` button below to test out sample data against the summarization REST API.
    ---
    tags:
      - name: Text Summarization
    parameters:
      - name: text
        in: query
        type: string
        required: true
        description: Text to be summarized
        default: 'In order to find the most relevant sentences in text, a graph is constructed where the vertices of the graph represent each sentence in a document and the edges between sentences are based on content overlap, namely by calculating the number of words that 2 sentences have in common. Based on this network of sentences, the sentences are fed into the Pagerank algorithm which identifies the most important sentences. When we want to extract a summary of the text, we can now take only the most important sentences. In order to find relevant keywords, the textrank algorithm constructs a word network. This network is constructed by looking which words follow one another. A link is set up between two words if they follow one another, the link gets a higher weight if these 2 words occur more frequenctly next to each other in the text. On top of the resulting network the Pagerank algorithm is applied to get the importance of each word. The top 1/3 of all these words are kept and are considered relevant. After this, a keywords table is constructed by combining the relevant words together if they appear following one another in the text.'
      - name: ratio
        in: query
        type: number
        required: true
        description: Percent of original text
        default: 0.20
    responses:
        200:
            description: The output values of the request
    """

    # Return GET output if called directly from the REST API.
    if request.method == 'GET':
        text = request.args.get("text")
        ratio = request.args.get("ratio")
        summary_dict = summarizer.smart_summary(
            text=text,
            ratio=float(ratio))
        return jsonify(summary_dict)
    elif request.method == 'POST':
        text = request.json.get("text")
        ratio = request.json.get("ratio")
        summary_dict = summarizer.smart_summary(
            text=text,
            ratio=float(ratio))
        return jsonify(summary_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
