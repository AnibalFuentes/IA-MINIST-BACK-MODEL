from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
from PIL import Image
import io
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permitir CORS para la app de Flutter

class DigitPredictor:
    def __init__(self, model_path='digit_recognition_model.h5'):
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Carga el modelo entrenado"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Modelo cargado exitosamente desde {model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise e
    
    def preprocess_image(self, image_data):
        """Preprocesa la imagen para el modelo"""
        try:
            # Decodificar imagen base64
            if isinstance(image_data, str):
                # Remover prefijo de data URL si existe
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # Decodificar base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = image_data
            
            # Convertir a escala de grises si es necesario
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convertir PIL a numpy array
            img_array = np.array(image)
            
            # Redimensionar a 28x28
            img_resized = cv2.resize(img_array, (28, 28))
            
            # Invertir colores si es necesario (MNIST tiene fondo negro, dígito blanco)
            # Asumimos que el usuario dibuja en negro sobre fondo blanco
            img_inverted = 255 - img_resized
            
            # Normalizar (0-255 -> 0-1)
            img_normalized = img_inverted.astype('float32') / 255.0
            
            # Reshape para el modelo (batch_size, height, width, channels)
            img_final = img_normalized.reshape(1, 28, 28, 1)
            
            return img_final
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise e
    
    def predict_digit(self, processed_image):
        """Predice el dígito"""
        try:
            # Hacer predicción
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Obtener probabilidades y dígito predicho
            probabilities = prediction[0]
            predicted_digit = np.argmax(probabilities)
            confidence = float(probabilities[predicted_digit])
            
            # Crear respuesta con todas las probabilidades
            result = {
                'predicted_digit': int(predicted_digit),
                'confidence': confidence,
                'probabilities': {str(i): float(prob) for i, prob in enumerate(probabilities)}
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise e

# Instancia global del predictor
predictor = None

@app.before_request
def initialize_predictor():
    """Inicializa el predictor antes de la primera request"""
    global predictor
    if predictor is None:
        try:
            predictor = DigitPredictor()
            logger.info("Predictor inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando predictor: {e}")
            raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servidor"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None and predictor.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal para predicción de dígitos"""
    try:
        # Verificar que el predictor esté inicializado
        if predictor is None or predictor.model is None:
            return jsonify({'error': 'Modelo no cargado'}), 500
        
        # Obtener datos de la request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        # Preprocesar imagen
        image_data = data['image']
        processed_image = predictor.preprocess_image(image_data)
        
        # Hacer predicción
        result = predictor.predict_digit(processed_image)
        
        # Agregar información adicional
        result['status'] = 'success'
        result['message'] = f'Dígito predicho: {result["predicted_digit"]} (confianza: {result["confidence"]:.2%})'
        
        logger.info(f"Predicción exitosa: {result['predicted_digit']} con confianza {result['confidence']:.2%}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({
            'error': 'Error procesando imagen',
            'details': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Endpoint para predicción de múltiples imágenes"""
    try:
        if predictor is None or predictor.model is None:
            return jsonify({'error': 'Modelo no cargado'}), 500
        
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({'error': 'No se proporcionaron imágenes'}), 400
        
        results = []
        
        for i, image_data in enumerate(data['images']):
            try:
                processed_image = predictor.preprocess_image(image_data)
                result = predictor.predict_digit(processed_image)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'image_index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        return jsonify({
            'error': 'Error procesando imágenes',
            'details': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    # Inicializar predictor manualmente para desarrollo
    try:
        predictor = DigitPredictor()
        logger.info("Predictor inicializado en modo desarrollo")
    except Exception as e:
        logger.error(f"Error inicializando predictor: {e}")
        exit(1)
    
    # Ejecutar servidor
    app.run(host='0.0.0.0', port=5000, debug=True)