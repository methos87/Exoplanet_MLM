# Kepler Exobolygók azonosítása Machine Learning segítségével

## Description
This project focuses on identifying exoplanets from Kepler's dataset using various machine learning techniques. The goal is to classify whether a celestial body is a confirmed exoplanet, a candidate, or a false positive. Advanced models such as Random Forest, Gradient Boosting, LightGBM, and Neural Networks are applied to predict the status of exoplanets based on observed features from the Kepler dataset.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd kepler-exobolygok-ml
   ```

2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the necessary datasets. Place the Kepler dataset (`cumulative_2024.09.03_11.45.57.csv`) in the `original_dataset` folder.
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
## Usage
1. **Run the Rest API:**
   - Run the training script to preprocess the data and train the machine learning models.
   ```bash
   gunicorn -w 4 -b 127.0.0.1:5000 --timeout 120 exobolygo_app:app
   ```

2. **Making Predictions:**
   - Use the prediction API to classify new data. Ensure the Flask application is running:
   - You can send POST requests with the JSON payload to make predictions about exoplanets.

3. **Testing:**
   - To run tests and ensure the model performs correctly, execute the following:
   ```bash
   pytest test/train_test.py
   ``

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
