Instalar Flask y otras dependencias: Si no tienes Flask instalado, instala las librerías necesarias. Abre tu terminal o línea de comandos y ejecuta:

```bash
pip install flask tensorflow scikit-learn joblib
```

```
├── app.py               # Archivo principal Flask
├── model.keras           # Tu modelo entrenado
└── scaler.joblib         # El escalador guardado
```

### Explicación del código:
Cargar el modelo y los escaladores:

Usamos tf.keras.models.load_model('model.keras') para cargar el modelo entrenado.

Usamos joblib.load('scaler_x.joblib') y joblib.load('scaler_y.joblib') para cargar los escaladores para X y y respectivamente.

Ruta /predict:

En esta ruta se espera una solicitud POST que contenga las características para hacer la predicción.

Las características son extraídas del JSON que el cliente envíe.

Las características son escaladas utilizando el escalador scaler_X.

Luego, el modelo realiza la predicción.

El valor predicho está en la escala logarítmica, por lo que deshacemos la transformación logarítmica usando scaler_y.inverse_transform(prediction_log).

Finalmente, devolvemos el precio predicho en formato JSON.

Ruta /:

Esta es una ruta de bienvenida simple para verificar que la aplicación Flask está funcionando correctamente.

Cómo hacer una solicitud POST a la API:
Para probar la API, puedes usar herramientas como Postman o hacer una solicitud POST con Python usando la librería requests.

Ejemplo de cómo hacer una solicitud POST con Python:

```python
import requests

# Datos de entrada (tus características de prueba)
data = {
    "features": [5000, 4, 3, 2, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1000, 200, 3]  # Ejemplo de características
}

# Hacer la solicitud POST
response = requests.post('http://127.0.0.1:5000/predict', json=data)

# Obtener la respuesta JSON
prediction = response.json()

print(f"Predicción de precio: {prediction['predicted_price']}")
```

Iniciar la aplicación Flask:
Para ejecutar tu servidor Flask, abre una terminal, navega a la carpeta donde se encuentra tu archivo app.py y ejecuta el siguiente comando:
```bash
python app.py
```

Esto iniciará el servidor Flask en http://127.0.0.1:5000/.

Probar la API:
Abre tu navegador y visita http://127.0.0.1:5000/ para ver la respuesta de bienvenida.

Usa una herramienta como Postman o el código en Python proporcionado para enviar una solicitud POST a http://127.0.0.1:5000/predict con las características necesarias y obtener la predicción.

![img](img\img_prediction.png)
![img](img\evaluation.png)