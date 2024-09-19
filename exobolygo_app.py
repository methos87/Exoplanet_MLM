import os
import pandas as pd
from exobolygo_model import MLModel
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
from flask_swagger_ui import get_swaggerui_blueprint

# Flask alkalmazás létrehozása
app = Flask(__name__)

# API létrehozása verziószámmal és címkével
api = Api(app, version='1.0', title='API Documentation')

# Swagger felület URL-jeinek megadása
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'

# Swagger UI létrehozása a megadott beállításokkal
swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "My API"
    }
)

# Swagger felület regisztrálása
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

# MLModel objektum létrehozása
obj_mlmodel = MLModel()

# PredictModel létrehozása az előrejelzéshez szükséges adatsorral
predict_model = api.model('PredictModel', {
    'inference_row': fields.List(fields.Raw, required=True,
                                 description='Egy adatsor előrejelzéshez')
})

# Fájl feltöltéshez szükséges paraméterek hozzáadása
file_upload = api.parser()
file_upload.add_argument('file', location='files',
                         type=FileStorage, required=True,
                         help='CSV fájl a modell betanításához')

# API namespace definiálása modell műveletekhez
ns = api.namespace('model', description='Modell műveletek')


# Új útvonal a modell betanításához
@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        # A feltöltött fájl paramétereinek elemzése
        args = file_upload.parse_args()
        uploaded_file = args['file']

        # Ellenőrzés, hogy a fájl CSV típusú-e
        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'Hiba: Érvénytelen fájltípus'}, 400

        # Fájl mentése ideiglenes útvonalra
        data_path = 'data/temp_dataset.csv'
        uploaded_file.save(data_path)

        try:
            # Az oszlopok beolvasása, valamint az adatok előfeldolgozása
            exo_column_names = pd.read_csv(data_path,
                                           on_bad_lines="skip")
            exo_data_clean = pd.read_csv(data_path,
                                         skiprows=143,
                                         low_memory=False,
                                         header=0)
            # Adat előfeldolgozási folyamat
            mydataframe = obj_mlmodel.preprocessing_pipeline(exo_column_names, exo_data_clean)
            print(mydataframe.head())

            # Modell betanítása és mentése
            train_accuracy, test_accuracy, lgbm = obj_mlmodel.train_and_save_model(mydataframe)
            obj_mlmodel.save_model(lgbm, 'src/models/lgbm_model.pkl')

            # A tisztított dataframe mentése és az ideiglenes fájl törlése
            mydataframe.to_csv('data/saved_dataframe_new.csv', index=False)
            os.remove(data_path)

            return {'üzenet': 'A modell sikeresen betanítva',
                    'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}, 200

        except Exception as e:
            # Hiba esetén belső szerverhiba üzenet
            return {'Üzenet': 'Belső szerverhiba', 'Hiba': str(e)}, 500


# Új útvonal az előrejelzéshez
@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
        try:
            # Bemeneti adatok lekérése
            data = request.get_json()

            # Státuszok leképezése
            status_mapping = {
                0: 'FALSE POSITIVE',
                1: 'CANDIDATE',
                2: 'CONFIRMED'
            }

            # Ellenőrzés, hogy van-e 'inference_row' a kérésben
            if 'inference_row' not in data:
                return {'hiba': 'Nincs inference_row megadva'}, 400

            # A bemeneti adatok normalizálása és előfeldolgozása
            inference_array = pd.json_normalize(data['inference_row'])
            df = obj_mlmodel.preprocessing_pipeline_inference(inference_array)

            # Előrejelzés futtatása a modell segítségével
            y_pred = obj_mlmodel.model.predict(df)
            y_pred_list = y_pred.tolist()

            # Előrejelzett értékek státuszokhoz rendelése
            y_pred_status = [status_mapping[pred] for pred in y_pred_list]

            # Sikeres előrejelzés visszaadása
            return {'üzenet': 'Előrejelzés sikeres', 'predikciósudo ': y_pred_status}, 200

        except Exception as e:
            # Hiba esetén belső szerverhiba üzenet
            return {'üzenet': 'Belső szerverhiba', 'hiba': str(e)}, 500


# Az alkalmazás futtatása
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
