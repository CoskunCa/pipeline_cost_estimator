from flask import Flask, render_template, request
import pickle
import numpy as np
import locale

# Yerel formatlama için Türkçe ayar
locale.setlocale(locale.LC_ALL, '')

app = Flask(__name__)

# Modelleri ve MAPE değerlerini yükle
with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('models/mape_values.pkl', 'rb') as f:
    mape_values = pickle.load(f)  # {'rf_mape': X, 'xgb_mape': Y, 'knn_mape': Z}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Formdan gelen verileri al
    diameter = float(request.form['diameter'])
    length = float(request.form['length'])
    line_valve = int(request.form['line_valve'])
    take_off_valve = int(request.form['take_off_valve'])
    pigging_station = int(request.form['pigging_station'])

    # Girdileri numpy array'e çevir
    input_data = np.array([[diameter, length, line_valve, take_off_valve, pigging_station]])

    # Modellerden tahmin al
    rf_prediction = rf_model.predict(input_data)[0]
    xgb_prediction = xgb_model.predict(input_data)[0]
    knn_prediction = knn_model.predict(input_data)[0]

    # MAPE değerlerini al
    rf_mape = mape_values['rf_mape']
    xgb_mape = mape_values['xgb_mape']
    knn_mape = mape_values['knn_mape']

    # Hata aralıkları ve formatlama
    rf_lower = rf_prediction * (1 - rf_mape / 100)
    rf_upper = rf_prediction * (1 + rf_mape / 100)

    xgb_lower = xgb_prediction * (1 - xgb_mape / 100)
    xgb_upper = xgb_prediction * (1 + xgb_mape / 100)

    knn_lower = knn_prediction * (1 - knn_mape / 100)
    knn_upper = knn_prediction * (1 + knn_mape / 100)

    # Sonuçları HTML'ye geçir
    results = f"""
    <p>Random Forest Tahmini: {locale.format_string('%.2f', rf_prediction, grouping=True)} $ (± %{rf_mape:.2f}) 
    ({locale.format_string('%.2f', rf_lower, grouping=True)} - {locale.format_string('%.2f', rf_upper, grouping=True)})</p>
    <p>XGBoost Tahmini: {locale.format_string('%.2f', xgb_prediction, grouping=True)} $ (± %{xgb_mape:.2f}) 
    ({locale.format_string('%.2f', xgb_lower, grouping=True)} - {locale.format_string('%.2f', xgb_upper, grouping=True)})</p>
    <p>KNN Tahmini: {locale.format_string('%.2f', knn_prediction, grouping=True)} $ (± %{knn_mape:.2f}) 
    ({locale.format_string('%.2f', knn_lower, grouping=True)} - {locale.format_string('%.2f', knn_upper, grouping=True)})</p>
    """

    return render_template(
        'index.html',
        results=results,
        diameter=diameter,
        length=length,
        line_valve=line_valve,
        take_off_valve=take_off_valve,
        pigging_station=pigging_station
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
