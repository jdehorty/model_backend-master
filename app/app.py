# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from joblib import load

model = load('model.joblib')

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
            "title": "Swagger API - Bentley Assignment",
            "version": "3.0",
            "description": "OpenAPI 3.0 Specification for the REST-API backend of the Amazon EC2 Hosted ML Model",
        },
        "basePath": "/",
        "schemes": [
            "http",
            "https"
        ]
    }
    return Swagger(app_, template=template)


create_swagger(app)


def prepare_employee_data(building, expertise, institute, num_conferences, num_postdocs, num_publications, num_reports,
                          status, title):
    # collect the incoming data
    num_conferences = int(num_conferences)
    is_fulltime = 1 if status == 'Fulltime' else 0
    postdocs = int(num_postdocs)
    reports = int(num_reports)
    publications = int(num_publications)
    # one_hot_encode
    building_10 = 1
    building_14 = 0
    building_Unknown = 0
    building_nan = 0
    if building[0] == 'Building 10':
        building_10 = 1
    if building[0] == 'Building 14':
        building_14 = 1
    # expertise
    expertise_infectious_diseases = 0
    expertise_bioinformatics = 0
    expertise_nan = 0
    if expertise == 'Infectious Diseases':
        expertise_infectious_diseases = 0
    if expertise == 'Bioinformatics':
        expertise_bioinformatics = 0
    # title
    title_Dr = 0
    title_Miss = 0
    title_Mr = 0
    title_Mrs = 0
    title_Ms = 0
    title_Unknown = 0
    title_nan = 0
    if title == 'Dr':
        title_Dr = 1
    if title == 'Miss':
        title_Miss = 1
    if title == 'Mr':
        title_Mr = 1
    if title == 'Mrs':
        title_Mrs = 1
    if title == 'Ms':
        title_Ms = 1
    if title == 'Unknown':
        title_Unknown = 1
    # institute
    institute_nci = 0
    institute_nei = 0
    institute_nhlbi = 0
    institute_nhgri = 0
    institute_nia = 0
    institute_niaid = 0
    institute_nimh = 0
    institute_Unknown = 0
    institute_nan = 0
    if institute == 'NCI':
        institute_nci = 1
    if institute == 'NEI':
        institute_nei = 1
    if institute == 'NHLBI':
        institute_nhlbi = 1
    if institute == 'NHGRI':
        institute_nhgri = 1
    if institute == 'NIA':
        institute_nia = 1
    if institute == 'NIAID':
        institute_niaid = 1
    if institute == 'NIMH':
        institute_nimh = 1
    if institute == 'Unknown':
        institute_Unknown = 1
    # predict
    user_designed_employee = [[num_conferences, postdocs, reports, publications, is_fulltime,
                               expertise_infectious_diseases, expertise_bioinformatics, expertise_nan,
                               institute_nci, institute_nei, institute_nhlbi, institute_nhgri, institute_nia,
                               institute_niaid, institute_nimh, institute_Unknown, institute_nan, building_10,
                               building_14, building_Unknown, building_nan, title_Dr, title_Miss, title_Mr,
                               title_Mrs, title_Ms, title_Unknown, title_nan]]
    return user_designed_employee


@app.route('/')
def index():
    from flask import request
    return f"Please refer to the <a href=\"{request.url_root}apidocs\">API documentation</a> for more details."


@app.route('/api/predict', methods=["GET", "POST"])
def send_json_to_requester():
    """Click here to test out sample data.
    Press the `Try it out` button below to test out sample data against the backend REST API.

    *Notes:*

    **GET:** Used in this demo only; OpenAPI 3.0 standards no longer supports body requests for GET. Returns: `STRING`

    **POST:** Used by the front-end service; will pass in parameters via POST. Returns: `JSON`
    ---
    tags:
      - name: Test Prediction
    parameters:
      - name: building
        in: query
        type: string
        required: true
        description: Name of a Building on the Main NIH Campus that is the location of work for an employee.
        default: 'Building 10'
      - name: num_publications
        in: query
        type: number
        required: true
        description: The total amount of peer-reviewed publications held by an employee.
        default: 20
      - name: num_conferences
        in: query
        type: number
        required: true
        description: The total amount of NIH-sanctioned conferences attended by an employee.
        default: 30
      - name: status
        in: query
        type: string
        required: true
        description: The classification of an employee as either 'Parttime' or 'Fulltime' (40 hours/week).
        default: 'Fulltime'
      - name: title
        in: query
        type: string
        required: true
        description: The official salutation of an employee; may correspond to educational level.
        default: 'Dr'
      - name: expertise
        in: query
        type: string
        required: true
        description: The specific field of expertise held by an employee.
        default: 'Infectious Diseases'
      - name: institute
        in: query
        type: string
        required: true
        description: One of the 27 Institutes of the National Institutes of Health where the employee works.
        default: 'NHLBI'
      - name: num_postdocs
        in: query
        type: number
        required: true
        description: The total amount of Postdoctoral Fellows that an employee oversees.
        default: 3
      - name: num_reports
        in: query
        type: number
        required: true
        description: The total amount of individuals (contractors and Federal workers) overseen by the employee.
        default: 7
    responses:
        200:
            description: The output values of the request
    """

    # Check to see if the request is coming from the front end
    try:
        building = request.json.get("building")
        num_publications = request.json.get("num_publications")
        num_conferences = request.json.get("num_conferences")
        status = request.json.get("status")
        title = request.json.get("title")
        expertise = request.json.get("expertise")
        institute = request.json.get("institute")
        num_postdocs = request.json.get("num_postdocs")
        num_reports = request.json.get("num_reports")

        # This method will put the data into the correct format for the model
        user_designed_employee = prepare_employee_data(building, expertise, institute, num_conferences, num_postdocs,
                                                       num_publications, num_reports, status, title)
        # model is already in memory from earlier
        y_pred = model.predict_proba(user_designed_employee)
        perc = y_pred[0][1] * 100
        return jsonify(perc)

    # Check to see if request is coming from Swagger API
    except AttributeError:
        # Collect data from incoming request
        building = request.args.get("building")
        num_publications = request.args.get("num_publications")
        num_conferences = request.args.get("num_conferences")
        status = request.args.get("status")
        title = request.args.get("title")
        expertise = request.args.get("expertise")
        institute = request.args.get("institute")
        num_postdocs = request.args.get("num_postdocs")
        num_reports = request.args.get("num_reports")
    
        # This method will put the data into the correct format for the model
        user_designed_employee = prepare_employee_data(building, expertise, institute, num_conferences, num_postdocs,
                                                       num_publications, num_reports, status, title)
        # model is already in memory from earlier
        y_pred = model.predict_proba(user_designed_employee)
        perc = y_pred[0][1] * 100
    
        # Return GET output if called directly from the REST API.
        if request.method == 'GET':
            return f"There\'s a {str(round(perc, 2))}% chance that the employee will work onsite."
    
        # Return POST output as if it were called from the front end.
        elif request.method == 'POST':
            return jsonify(perc)
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
