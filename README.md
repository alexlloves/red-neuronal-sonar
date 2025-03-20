# Sonar Classification Model

Este proyecto contiene un modelo de red neuronal entrenado para clasificar datos del conjunto de datos **Sonar** utilizando PyTorch. El modelo tiene como objetivo predecir si los datos representan una roca o un mineral a partir de 60 características de entrada, las cuales provienen de un radar sonoro que detecta rocas y minerales en un entorno marino.

## Descripción

El objetivo de este proyecto es implementar y entrenar un modelo de red neuronal para realizar clasificación en el conjunto de datos Sonar, que es utilizado para clasificar objetos detectados por un sonar. El modelo está basado en una red neuronal de 3 capas y se ha entrenado utilizando PyTorch.

### Arquitectura del Modelo

- **Capa 1**: Capa densa con 50 unidades (neuronas).
- **Capa 2**: Capa densa con 50 unidades (neuronas).
- **Capa 3**: Capa de salida con 2 unidades (para clasificación binaria).

Las funciones de activación ReLU se utilizan en las dos primeras capas y Softmax en la capa de salida para convertir las predicciones en probabilidades.

## Instalación

Para utilizar este proyecto, asegúrate de tener Python 3.x y las siguientes bibliotecas instaladas:

- **PyTorch**: para crear y entrenar el modelo de red neuronal.
- **Pandas**: para la manipulación de datos.
- **Scikit-learn**: para la normalización de características.