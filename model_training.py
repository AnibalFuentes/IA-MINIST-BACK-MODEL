import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pickle

class DigitRecognitionModel:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Carga y preprocesa los datos MNIST"""
        print("Cargando datos MNIST...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalizar los datos (0-255 -> 0-1)
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape para CNN (añadir canal de color)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # Convertir etiquetas a categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"Datos de entrenamiento: {x_train.shape}")
        print(f"Datos de prueba: {x_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def create_model(self):
        """Crea la arquitectura de la red neuronal convolucional"""
        model = keras.Sequential([
            # Primera capa convolucional
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Segunda capa convolucional
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Tercera capa convolucional
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten y capas densas
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        print("Arquitectura del modelo:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
        """Entrena el modelo"""
        print("Iniciando entrenamiento...")
        
        # Callbacks para mejorar el entrenamiento
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """Evalúa el modelo"""
        print("Evaluando modelo...")
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Precisión en datos de prueba: {test_acc:.4f}")
        
        # Predicciones para reporte detallado
        predictions = self.model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\nReporte de clasificación:")
        print(classification_report(y_true, y_pred))
        
        return test_acc
    
    def save_model(self, filepath='digit_recognition_model.h5'):
        """Guarda el modelo entrenado"""
        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def plot_training_history(self):
        """Visualiza la historia del entrenamiento"""
        if self.history is None:
            print("No hay historia de entrenamiento para mostrar")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Precisión
        ax1.plot(self.history.history['accuracy'], label='Entrenamiento')
        ax1.plot(self.history.history['val_accuracy'], label='Validación')
        ax1.set_title('Precisión del Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Precisión')
        ax1.legend()
        
        # Pérdida
        ax2.plot(self.history.history['loss'], label='Entrenamiento')
        ax2.plot(self.history.history['val_loss'], label='Validación')
        ax2.set_title('Pérdida del Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Función principal para entrenar el modelo"""
    # Crear instancia del modelo
    digit_model = DigitRecognitionModel()
    
    # Cargar y preprocesar datos
    (x_train, y_train), (x_test, y_test) = digit_model.load_and_preprocess_data()
    
    # Crear modelo
    digit_model.create_model()
    
    # Entrenar modelo
    digit_model.train_model(x_train, y_train, x_test, y_test, epochs=15)
    
    # Evaluar modelo
    digit_model.evaluate_model(x_test, y_test)
    
    # Guardar modelo
    digit_model.save_model('digit_recognition_model.h5')
    
    # Mostrar gráficos de entrenamiento
    digit_model.plot_training_history()
    
    print("¡Entrenamiento completado!")

if __name__ == "__main__":
    main()