# Kepler Exobolygók azonosítása Machine Learning segítségével

## Description

This project focuses on identifying exoplanets from Kepler's dataset using various machine learning techniques. The goal
is to classify whether a celestial body is a confirmed exoplanet, a candidate, or a false positive. Advanced models such
as Random Forest, Gradient Boosting, LightGBM, and Neural Networks are applied to predict the status of exoplanets based
on observed features from the Kepler dataset.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/methos87/Exoplanet_MLM.git
   cd Exoplanet_MLM
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary datasets. Place the Kepler dataset (`cumulative_2024.09.03_11.45.57.csv`) in the
   `original_dataset` folder.
   https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative

## Jupyter Notebook
1. You can find the original Jupyter Notebook:
   ```bash
   /orginal_notbook/exobolygo.ipynb
   ```
## Usage

1. **Run the Rest API:**
    - Run the training script to preprocess the data and train the machine learning models.
   ```bash
   gunicorn -w 4 -b 127.0.0.1:5000 --timeout 120 exobolygo_app:app
   ```
   
2. **Training the model:**
    - Use the prediction API to classify new data. Ensure the Flask application is running:
    - Upload the dataset (`cumulative_2024.09.03_11.45.57.csv`) in the `original_dataset` folder to the training.
    ```bash
     'POST /model/train'
    ```
    

3. **Making Predictions:**
    - Use the prediction API to classify new data. Ensure the Flask application is running:
    - You can send POST requests with the JSON payload to make predictions about exoplanets.
    - Copy data from the 'json/sample_data.json' to the 'inference_row' -> {}
    ```bash
     {
      "inference_row": [
          {}
        ]
     }
    ```
   
3. **Testing:**
    - To run tests and ensure the model performs correctly, execute the following:
   ```bash
   pytest test/train_test.py
   ```

## Docker

1. **Run Rest API from docker container:**
    - Download the docker file.
    - Run the following command:
   ```bash
   sudo docker run -d -p 5000:5000 my-flask-api
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
