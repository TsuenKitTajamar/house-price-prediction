from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Cargar el modelo
model = tf.keras.models.load_model('files/best_model.keras')

# Página principal
@app.route('/')
def index():
    return render_template('index.html')

# Función para generar las características adicionales
def generate_features(data):
    area = data[0]
    bedrooms = data[1]
    bathrooms = data[2]
    stories = data[3]
    mainroad = data[4]
    guestroom = data[5]
    basement = data[6]
    hotwaterheating = data[7]
    airconditioning = data[8]
    parking = data[9]
    prefarea = data[10]
    furnishingstatus = data[11:14]  # Las tres características de mobiliario

    # Características generadas
    total_rooms = bedrooms + bathrooms
    area_per_room = area / total_rooms if total_rooms != 0 else 0
    bedrooms_per_area = bedrooms / area if area != 0 else 0
    bathrooms_per_area = bathrooms / area if area != 0 else 0
    bedrooms_bathrooms_ratio = bedrooms / bathrooms if bathrooms != 0 else 0
    bathrooms_per_bedroom = bathrooms / bedrooms if bedrooms != 0 else 0
    bedrooms_per_story = bedrooms * stories
    bathroom_stories = bathrooms * stories

    # Puntuaciones
    quality_score = (
        airconditioning * 3 +
        basement * 2 +
        guestroom * 1.5 +
        hotwaterheating * 2.5 +
        prefarea * 3
    )
    convenience_score = (
        mainroad * 2 +
        parking * 1.5 +
        prefarea * 2
    )
    comfort_score = (
        airconditioning * 2 +
        hotwaterheating * 1.5 +
        basement * 2
    )
    furnishing_score = (
        furnishingstatus[0] * 3 +
        furnishingstatus[1] * 2 +
        furnishingstatus[2] * 1
    )
    accessibility_score = (
        mainroad * 2 +
        area / 1000 +
        stories * 1.5
    )
    space_efficiency_score = (
        area / (bedrooms + bathrooms) * 1.5 +
        stories * 1
    )
    luxury_score = quality_score * convenience_score
    quality_per_room = quality_score / total_rooms if total_rooms != 0 else 0
    luxury_per_area = luxury_score / area * 1000 if area != 0 else 0

    # Devuelve la lista de características generadas
    return [
        total_rooms, area_per_room, bedrooms_per_area, bathrooms_per_area, bedrooms_bathrooms_ratio,
        bathrooms_per_bedroom, bedrooms_per_story, bathroom_stories, quality_score, convenience_score,
        comfort_score, furnishing_score, accessibility_score, space_efficiency_score, luxury_score,
        quality_per_room, luxury_per_area
    ]

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        data = [
            float(request.form['area']),
            float(request.form['bedrooms']),
            float(request.form['bathrooms']),
            float(request.form['stories']),
            int(request.form['mainroad']),
            int(request.form['guestroom']),
            int(request.form['basement']),
            int(request.form['hotwaterheating']),
            int(request.form['airconditioning']),
            float(request.form['parking']),
            int(request.form['prefarea']),
        ]

        # Manejo correcto del estado de mobiliario
        furnishingstatus = request.form['furnishingstatus']
        if furnishingstatus == "furnished":
            data.extend([1, 0, 0])
        elif furnishingstatus == "semi-furnished":
            data.extend([0, 1, 0])
        else:  # unfurnished
            data.extend([0, 0, 1])

        # Generar las características adicionales
        additional_features = generate_features(data)

        # Agregar las características generadas a los datos originales
        data.extend(additional_features)

        # Convertir a numpy array y predecir
        input_data = np.array([data])

        # Descalar el valor predicho y aplicar el logaritmo
        scaler_X = joblib.load('files/scaler_X.joblib')
        scaler_y = joblib.load('files/scaler_y.joblib')

        # Preprocesar los datos de entrada (como se hizo con los datos de entrenamiento)
        data_multiple_to_predict_scaled = scaler_X.transform(input_data)  # Usar el scaler_X entrenado

        # Realizar la predicción (en la escala logarítmica)
        log_predictions = model.predict(data_multiple_to_predict_scaled)

        # Desescalar la predicción de y (invertir MinMaxScaler)
        log_predictions = scaler_y.inverse_transform(log_predictions)

        # Aplicar la transformación inversa log1p para obtener el precio original
        predictions_original_scale = np.expm1(log_predictions)  # Inverso de log1p

        # Convertir el valor predicho a float
        price = float(predictions_original_scale[0][0])

        # Devolver la predicción como respuesta JSON
        return jsonify({'predicted_price': f"{price:,.2f}$"})

    
    except Exception as e:
        # Imprimir el error para ver detalles
        print(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
