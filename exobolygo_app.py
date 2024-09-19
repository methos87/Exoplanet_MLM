import os
import pandas as pd
from exobolygo_model import MLModel
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)
api = Api(app, version='1.0', title='API Documentation')

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "My API"
    }
)

app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

obj_mlmodel = MLModel()

predict_model = api.model('PredictModel', {
    'inference_row': fields.List(fields.Raw, required=True,
                                 description='A row of data for inference')
})

file_upload = api.parser()
file_upload.add_argument('file', location='files',
                         type=FileStorage, required=True,
                         help='CSV file for training')

ns = api.namespace('model', description='Model operations')


@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        uploaded_file = args['file']
        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'Error: Invalid file type'}, 400

        data_path = 'data/temp_dataset.csv'
        uploaded_file.save(data_path)

        try:
            exo_column_names = pd.read_csv(data_path,
                                           on_bad_lines="skip")
            exo_data_clean = pd.read_csv(data_path,
                                         skiprows=143,
                                         low_memory=False,
                                         header=0)
            mydataframe = obj_mlmodel.preprocessing_pipeline(exo_column_names, exo_data_clean)
            print(mydataframe.head())
            train_accuracy, test_accuracy, lgbm = obj_mlmodel.train_and_save_model(mydataframe)
            obj_mlmodel.save_model(lgbm, 'src/models/lgbm_model.pkl')
            mydataframe.to_csv('data/saved_dataframe_new.csv', index=False)
            os.remove(data_path)

            return {'message': 'Model Trained Successfully',
                    'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}, 200

        except Exception as e:
            return {'Message': 'Internal Server Error', 'Error': str(e)}, 500


@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
        try:
            data = request.get_json()

            # Check if 'inference_row' exists in the data
            if 'inference_row' not in data:
                return {'error': 'No inference_row found'}, 400

            inference_array= pd.json_normalize(data['inference_row'])
            print("inference_array:", inference_array)

            # Pass reshaped array to your preprocessing function
            df = obj_mlmodel.preprocessing_pipeline_inference(inference_array)
            print("inference df_ia:", df)

            # Get prediction
            y_pred = obj_mlmodel.model.predict(df)
            print("prediction:", y_pred)

            # Convert numpy array to list for JSON serialization
            y_pred_list = y_pred.tolist()

            # Return prediction result
            return {'message': 'Inference Successful', 'prediction': y_pred_list}, 200
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
