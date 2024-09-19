import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# A szülő könyvtár hozzáadása a sys.path-hez, így importálhatjuk az MLModel-t
current_file_path = Path(__file__).resolve()
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))

from exobolygo_model import MLModel


def test_prediction_accuracy():
    obj_mlmodel = MLModel()
    data_path = "original_dataset/cumulative_2024.09.03_11.45.57.csv"

    # Ellenőrizzük, hogy a fájl létezik-e
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    # Oszlopnevek beolvasása
    exo_column_names = pd.read_csv(data_path, on_bad_lines="skip")

    # Adatok tisztítása
    exo_data_clean = pd.read_csv(data_path, skiprows=143, low_memory=False, header=0)

    # Előfeldolgozási pipeline végrehajtása
    df_preprocessed = obj_mlmodel.preprocessing_pipeline(exo_column_names, exo_data_clean)

    # Céltartomány különválasztása
    y_expected = df_preprocessed["Exoplanet_Archive_Disposition_Encoded"]
    X_train = df_preprocessed.drop(columns="Exoplanet_Archive_Disposition_Encoded", axis=1)

    # Tanulási pipeline pontosságának ellenőrzése
    accuracy_train_pipeline_full = obj_mlmodel.get_accuracy_full(X_train, y_expected)
    accuracy_train_pipeline_full = np.round(accuracy_train_pipeline_full, 2)

    # Inferencia fázis
    preprocessed_list = obj_mlmodel.preprocessing_pipeline_inference(exo_data_clean)

    # Ellenőrizzük, hogy az inferencia pipeline is ugyanúgy van értékelve
    accuracy_inference_pipeline_full = obj_mlmodel.get_accuracy_full(preprocessed_list, y_expected)
    accuracy_inference_pipeline_full = np.round(accuracy_inference_pipeline_full, 2)

    print(accuracy_train_pipeline_full, accuracy_inference_pipeline_full)

    # Pontosság validálása
    assert accuracy_train_pipeline_full == accuracy_inference_pipeline_full, \
        "Az inferencia előrejelzési pontosság nem felel meg a vártnak"
