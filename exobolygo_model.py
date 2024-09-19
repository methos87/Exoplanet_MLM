import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from flask import jsonify
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MLModel:
    def __init__(self):
        self.exo_data = pd.DataFrame()
        self.model = (MLModel.load_model("src/models/lgbm_model.pkl")
                      if os.path.exists("src/models/lgbm_model.pkl")
                      else print("src/models/lgbm_model.pkl does not exist"))

    def predict(self, inference_row):
        try:
            infer_array = pd.Series(inference_row, dtype=str)
            df = self.preprocessing_pipeline_inference(infer_array)
            y_pred = self.model.predict(df)

            return jsonify({'prediction': y_pred.tolist()})

        except Exception as e:
            return jsonify({'message': 'Internal Server Error', 'error': str(e)}), 500

    def preprocessing_pipeline(self, exo_column_names, exo_data):

        folder = 'data/'
        MLModel.create_new_folder(folder)

        folder = 'src/models/'
        MLModel.create_new_folder(folder)

        folder = 'json/'
        MLModel.create_new_folder(folder)

        exo_data.to_csv("data/exo_data.csv")

        exo_column_names = exo_column_names.iloc[:-1, :]
        exo_column_names = exo_column_names.iloc[2:, :]

        column_names = dict()
        for index, row in exo_column_names.iterrows():
            temp = row.values[0][9:].split(": ")
            column_names[temp[0].lstrip()] = (temp[1]
                                              .lstrip()
                                              .replace(' ', '_')
                                              .replace('[', '')
                                              .replace(']', '')
                                              .replace('.', ''))

        # Convert and write JSON object to file
        with open("json/columns_names.json", "w") as outfile:
            json.dump(column_names, outfile)

        with open('json/data_types.json', 'r') as file:
            data_types = json.load(file)

        self.exo_data = pd.read_csv(r"data/exo_data.csv",
                                    low_memory=False,
                                    skip_blank_lines=True,
                                    header=1,
                                    dtype=data_types)

        sample_data = self.exo_data.loc[0]
        sample_data.to_json("json/sample_data.json", orient='index')

        self.exo_data = self.exo_data.iloc[1:]

        with open('json/columns_names.json', 'r') as file:
            column_names = json.load(file)

        self.exo_data = self.exo_data.rename(columns=column_names, inplace=False)

        for column in self.exo_data.columns:
            if not self.exo_data[column].any():
                self.exo_data = self.exo_data.drop(column, axis=1)

        drop_names = ["Vetting_Status",
                      "rowid",
                      "Date_of_Last_Parameter_Update",
                      "Disposition_Using_Kepler_Data",
                      "Disposition_Provenance",
                      "Link_to_DV_Report",
                      "Link_to_DV_Summary",
                      "KOI_Name",
                      "Kepler_Name",
                      "Disposition_Score"]

        self.exo_data = self.exo_data.drop(drop_names, axis=1)

        self.exo_data.to_csv(r"data/exo_data_new.csv")

        self.exo_data.Exoplanet_Archive_Disposition.value_counts()

        categorical = self.exo_data.select_dtypes(include=[object])

        le = LabelEncoder()

        for column in ['Exoplanet_Archive_Disposition', 'Comment', 'Planetary_Fit_Type',
                       'Limb_Darkening_Model', 'Parameters_Provenance', 'TCE_Delivery',
                       'Quarters', 'Transit_Model', 'Stellar_Parameter_Provenance']:
            self.exo_data[column + "_Encoded"] = le.fit_transform(self.exo_data[column])

        nulls = self.exo_data.isnull().sum()
        count = 1

        for key, value in nulls.items():
            # print(f"{count}  {key}{(70 - len(key)) * ' '}{value}")
            count += 1

        self.exo_data.drop(categorical.columns, inplace=True, axis=1)

        self.exo_data.dropna(subset=["Orbital_Period_Upper_Unc_days"], inplace=True)

        self.exo_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.exo_data = self.create_weight(self.exo_data,
                                           "Orbital_Period_days",
                                           "Orbital_Period_Upper_Unc_days",
                                           "Orbital_Period_Lower_Unc_days")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Transit_Epoch_BKJD",
                                           "Transit_Epoch_Upper_Unc_BKJD",
                                           "Transit_Epoch_Lower_Unc_BKJD")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Transit_Epoch_BJD",
                                           "Transit_Epoch_Upper_Unc_BJD",
                                           "Transit_Epoch_Lower_Unc_BJD")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Impact_Parameter",
                                           "Impact_Parameter_Upper_Unc",
                                           "Impact_Parameter_Lower_Unc")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Transit_Duration_hrs",
                                           "Transit_Duration_Upper_Unc_hrs",
                                           "Transit_Duration_Lower_Unc_hrs")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Transit_Depth_ppm",
                                           "Transit_Depth_Upper_Unc_ppm",
                                           "Transit_Depth_Lower_Unc_ppm")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Planet-Star_Radius_Ratio",
                                           "Planet-Star_Radius_Ratio_Upper_Unc",
                                           "Planet-Star_Radius_Ratio_Lower_Unc")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Fitted_Stellar_Density_g/cm**3",
                                           "Fitted_Stellar_Density_Upper_Unc_g/cm**3",
                                           "Fitted_Stellar_Density_Lower_Unc_g/cm**3")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Planetary_Radius_Earth_radii",
                                           "Planetary_Radius_Upper_Unc_Earth_radii",
                                           "Planetary_Radius_Lower_Unc_Earth_radii")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Insolation_Flux_Earth_flux",
                                           "Insolation_Flux_Upper_Unc_Earth_flux",
                                           "Insolation_Flux_Lower_Unc_Earth_flux")
        self.exo_data = self.create_weight(self.exo_data,
                                           "Planet-Star_Distance_over_Star_Radius",
                                           "Planet-Star_Distance_over_Star_Radius_Upper_Unc",
                                           "Planet-Star_Distance_over_Star_Radius_Lower_Unc")

        classification_features = ["Orbital_Period_days",
                                   "Orbital_Period_days_weight",
                                   "Transit_Epoch_BKJD",
                                   "Transit_Epoch_BKJD_weight",
                                   "Transit_Epoch_BJD",
                                   "Transit_Epoch_BJD_weight",
                                   "Impact_Parameter",
                                   "Impact_Parameter_weight",
                                   "Transit_Duration_hrs",
                                   "Transit_Duration_hrs_weight",
                                   "Transit_Depth_ppm",
                                   "Transit_Depth_ppm_weight",
                                   "Planet-Star_Radius_Ratio",
                                   "Planet-Star_Radius_Ratio_weight",
                                   "Fitted_Stellar_Density_g/cm**3",
                                   "Fitted_Stellar_Density_g/cm**3_weight",
                                   "Planetary_Radius_Earth_radii",
                                   "Planetary_Radius_Earth_radii_weight",
                                   "Orbit_Semi-Major_Axis_au",
                                   "Inclination_deg",
                                   "Equilibrium_Temperature_K",
                                   "Insolation_Flux_Earth_flux",
                                   "Insolation_Flux_Earth_flux_weight",
                                   "Planet-Star_Distance_over_Star_Radius",
                                   "Planet-Star_Distance_over_Star_Radius_weight",
                                   "Stellar_Effective_Temperature_K",
                                   "Stellar_Surface_Gravity_log10(cm/s**2)",
                                   "Stellar_Metallicity_dex",
                                   "Stellar_Radius_Solar_radii",
                                   "Stellar_Mass_Solar_mass",
                                   "Comment_Encoded",
                                   "Planetary_Fit_Type_Encoded",
                                   "Parameters_Provenance_Encoded",
                                   "TCE_Delivery_Encoded",
                                   "Quarters_Encoded",
                                   "Stellar_Parameter_Provenance_Encoded",
                                   "Exoplanet_Archive_Disposition_Encoded", ]

        new_exo_data = self.exo_data[classification_features]
        new_exo_data = new_exo_data.dropna()

        # Outliers
        z_scores = np.abs(stats.zscore(new_exo_data))
        threshold = 3
        outliers = (z_scores > threshold)
        outlier_indices = np.where(outliers.any(axis=1))[0]

        new_exo_data = new_exo_data.reset_index(drop=True)
        cleaned_data = new_exo_data.drop(outlier_indices, axis=0)

        # Apply log transformation to specific columns
        cleaned_data['Log_Orbital_Period'] = np.log(cleaned_data['Orbital_Period_days'] + 1)
        cleaned_data['Log_Transit_Depth'] = np.log(cleaned_data['Transit_Depth_ppm'] + 1)

        print("Preprocessing is completed")

        return cleaned_data

    def preprocessing_pipeline_inference(self, sample_data):

        sample_data.to_csv("data/sample_data.csv")

        with open('json/data_types.json', 'r') as file:
            data_types = json.load(file)

        if len(sample_data) > 1:
            count = 1
        else:
            count = 0

        sample = pd.read_csv(r"data/sample_data.csv",
                             low_memory=False,
                             skip_blank_lines=True,
                             header=count,
                             dtype=data_types)

        with open('json/columns_names.json', 'r') as file:
            column_names = json.load(file)

        sample = sample.rename(columns=column_names, inplace=False)

        if len(sample) > 1:
            sample = sample.iloc[1:]

        for column in sample.columns:
            if not sample[column].any():
                sample = sample.drop(column, axis=1)

        drop_names = ["Vetting_Status",
                      "rowid",
                      "Date_of_Last_Parameter_Update",
                      "Disposition_Using_Kepler_Data",
                      "Disposition_Provenance",
                      "Link_to_DV_Report",
                      "Link_to_DV_Summary",
                      "KOI_Name",
                      "Kepler_Name",
                      "Disposition_Score"]

        sample = sample.drop(drop_names, axis=1)

        categorical = sample.select_dtypes(include=[object])

        le = LabelEncoder()

        for column in ['Exoplanet_Archive_Disposition', 'Comment', 'Planetary_Fit_Type',
                       'Limb_Darkening_Model', 'Parameters_Provenance', 'TCE_Delivery',
                       'Quarters', 'Transit_Model', 'Stellar_Parameter_Provenance']:
            sample[column + "_Encoded"] = le.fit_transform(sample[column])

        nulls = sample.isnull().sum()
        count = 1

        for key, value in nulls.items():
            # print(f"{count}  {key}{(70 - len(key)) * ' '}{value}")
            count += 1

        sample.drop(categorical.columns, inplace=True, axis=1)

        sample.dropna(subset=["Orbital_Period_Upper_Unc_days"], inplace=True)

        sample.replace([np.inf, -np.inf], np.nan, inplace=True)

        sample = self.create_weight(sample,
                                    "Orbital_Period_days",
                                    "Orbital_Period_Upper_Unc_days",
                                    "Orbital_Period_Lower_Unc_days")
        sample = self.create_weight(sample,
                                    "Transit_Epoch_BKJD",
                                    "Transit_Epoch_Upper_Unc_BKJD",
                                    "Transit_Epoch_Lower_Unc_BKJD")
        sample = self.create_weight(sample,
                                    "Transit_Epoch_BJD",
                                    "Transit_Epoch_Upper_Unc_BJD",
                                    "Transit_Epoch_Lower_Unc_BJD")
        sample = self.create_weight(sample,
                                    "Impact_Parameter",
                                    "Impact_Parameter_Upper_Unc",
                                    "Impact_Parameter_Lower_Unc")
        sample = self.create_weight(sample,
                                    "Transit_Duration_hrs",
                                    "Transit_Duration_Upper_Unc_hrs",
                                    "Transit_Duration_Lower_Unc_hrs")
        sample = self.create_weight(sample,
                                    "Transit_Depth_ppm",
                                    "Transit_Depth_Upper_Unc_ppm",
                                    "Transit_Depth_Lower_Unc_ppm")
        sample = self.create_weight(sample,
                                    "Planet-Star_Radius_Ratio",
                                    "Planet-Star_Radius_Ratio_Upper_Unc",
                                    "Planet-Star_Radius_Ratio_Lower_Unc")
        sample = self.create_weight(sample,
                                    "Fitted_Stellar_Density_g/cm**3",
                                    "Fitted_Stellar_Density_Upper_Unc_g/cm**3",
                                    "Fitted_Stellar_Density_Lower_Unc_g/cm**3")
        sample = self.create_weight(sample,
                                    "Planetary_Radius_Earth_radii",
                                    "Planetary_Radius_Upper_Unc_Earth_radii",
                                    "Planetary_Radius_Lower_Unc_Earth_radii")
        sample = self.create_weight(sample,
                                    "Insolation_Flux_Earth_flux",
                                    "Insolation_Flux_Upper_Unc_Earth_flux",
                                    "Insolation_Flux_Lower_Unc_Earth_flux")
        sample = self.create_weight(sample,
                                    "Planet-Star_Distance_over_Star_Radius",
                                    "Planet-Star_Distance_over_Star_Radius_Upper_Unc",
                                    "Planet-Star_Distance_over_Star_Radius_Lower_Unc")

        classification_features = ["Orbital_Period_days",
                                   "Orbital_Period_days_weight",
                                   "Transit_Epoch_BKJD",
                                   "Transit_Epoch_BKJD_weight",
                                   "Transit_Epoch_BJD",
                                   "Transit_Epoch_BJD_weight",
                                   "Impact_Parameter",
                                   "Impact_Parameter_weight",
                                   "Transit_Duration_hrs",
                                   "Transit_Duration_hrs_weight",
                                   "Transit_Depth_ppm",
                                   "Transit_Depth_ppm_weight",
                                   "Planet-Star_Radius_Ratio",
                                   "Planet-Star_Radius_Ratio_weight",
                                   "Fitted_Stellar_Density_g/cm**3",
                                   "Fitted_Stellar_Density_g/cm**3_weight",
                                   "Planetary_Radius_Earth_radii",
                                   "Planetary_Radius_Earth_radii_weight",
                                   "Orbit_Semi-Major_Axis_au",
                                   "Inclination_deg",
                                   "Equilibrium_Temperature_K",
                                   "Insolation_Flux_Earth_flux",
                                   "Insolation_Flux_Earth_flux_weight",
                                   "Planet-Star_Distance_over_Star_Radius",
                                   "Planet-Star_Distance_over_Star_Radius_weight",
                                   "Stellar_Effective_Temperature_K",
                                   "Stellar_Surface_Gravity_log10(cm/s**2)",
                                   "Stellar_Metallicity_dex",
                                   "Stellar_Radius_Solar_radii",
                                   "Stellar_Mass_Solar_mass",
                                   "Comment_Encoded",
                                   "Planetary_Fit_Type_Encoded",
                                   "Parameters_Provenance_Encoded",
                                   "TCE_Delivery_Encoded",
                                   "Quarters_Encoded",
                                   "Stellar_Parameter_Provenance_Encoded", ]

        sample = sample[classification_features]
        sample = sample.dropna()

        # Outliers
        z_scores = np.abs(stats.zscore(sample))
        threshold = 3
        outliers = (z_scores > threshold)
        outlier_indices = np.where(outliers.any(axis=1))[0]

        sample = sample.reset_index(drop=True)
        cleaned_sample = sample.drop(outlier_indices, axis=0)

        # Apply log transformation to specific columns
        cleaned_sample['Log_Orbital_Period'] = np.log(cleaned_sample['Orbital_Period_days'] + 1)
        cleaned_sample['Log_Transit_Depth'] = np.log(cleaned_sample['Transit_Depth_ppm'] + 1)

        print("Interference is completed")
        return cleaned_sample

    def get_accuracy(self, X_train, X_test, y_train, y_test):
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        return train_accuracy, test_accuracy

    def train_and_save_model(self, cleaned_data):

        y_KF = cleaned_data["Exoplanet_Archive_Disposition_Encoded"]
        X_KF = cleaned_data.drop(columns="Exoplanet_Archive_Disposition_Encoded", axis=1)

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_KF)
        X_scaled = pd.DataFrame(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_KF, test_size=0.33, random_state=500)
        train_accuracies = []
        cv_accuracies = []

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        kf = KFold(n_splits=5)
        kf.get_n_splits(X_train)

        for i, (train_index, cv_index) in enumerate(kf.split(X_train)):
            lgbm = LGBMClassifier(boosting_type='gbdt',
                                  colsample_bytree=1.0,
                                  learning_rate=0.1,
                                  max_depth=5,
                                  min_child_samples=10,
                                  n_estimators=100,
                                  num_leaves=20,
                                  objective='multiclass',
                                  subsample=1.0)
            lgbm.fit(X_train.iloc[train_index], y_train.iloc[train_index])
            self.model = lgbm
            ac = self.get_accuracy(X_train.iloc[train_index],
                                   X_train.iloc[cv_index],
                                   y_train[train_index],
                                   y_train[cv_index])
            train_accuracies.append(ac[0])
            cv_accuracies.append(ac[1])
        train_accuracy = np.mean(train_accuracies)
        cv_accuracy = np.mean(cv_accuracies)

        return train_accuracy, cv_accuracy, lgbm

    def get_accuracy_full(self, X, y):
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        return accuracy

    @staticmethod
    def create_weight(data, feature, feature_err1, feature_err2):
        # Calculate absolute and relative uncertainties
        data[feature + "_abs_uncertainty"] = np.abs(data[feature_err1] - data[feature_err2])

        # Prevent division by zero in the relative uncertainty calculation
        epsilon = 1e-6  # small value to avoid division by zero
        data[feature + "_rel_uncertainty"] = data[feature + "_abs_uncertainty"] / (
                data[feature] + epsilon)

        # Calculate weight and handle infinite values
        data[feature + "_weight"] = 1 / (data[feature + "_rel_uncertainty"] + epsilon)

        # Remove temporary columns for cleanliness
        data.drop(columns=[feature + "_abs_uncertainty", feature + "_rel_uncertainty"], inplace=True)

        return data

    @staticmethod
    def create_new_folder(folder):
        Path(folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_model(model, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
