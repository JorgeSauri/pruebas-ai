# Requerimientos de librerías:
#pip install transformers
#pip install transformers scipy ftfy accelerate

"""
  Usar DistilBERT para crear un modelo que a partir de un texto, intente inferir los valores nutricionales (kcal, proteina, carbs, grasas, etc.)
  1. Cargamos el dataframe que ya tenemos de alimentos con sus valores nutricionales, y generaremos al azar otro dataframe con conjuntos de elementos con cantidades aleatorias y calcularemos los valores nutricionales de cada receta ficticia.
  2. Tomaremos cada entrada del dataframe ficticio, tokenizanmos y creamos embbedings con DistilBERT, y normalizamos los valores nutricionales. Estos serán nuestros valores X_Data y Y_Data que servirán para entrenar nuestro modelo de regresión.
  3. Entrenaremos un modelo de regresión que aprenda de acuerdo a una matriz de emmbedings de tokens (utilizando el Tokenizer de DistilBERT), a inferir los valores nutricionales aproximados.

  Usaremos DistilBERT para hacer un modelo que aprenda a extraer el ingrediente principal de un texto
  * Esto lo haremos creando un dataset con un ingrediente, rodeado de palabras aleatorias, para que el modelo aprenda a identificar un alimento en una frase
    - Ejemplo:
               Cadena='12 rodajas de Manzana roja golden - marca X'
               Alimento = 'manzana'

"""

# Importar librerías
import pandas as pd
from transformers import TFDistilBertModel, DistilBertTokenizerFast
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


NUM_RECETAS = 3000
EMB_SIZE = 128
VOCAB_SIZE = 768


def feature_vector_similarity(feature_vec1, feature_vec2):
    # Calcular la similitud entre dos vectores de características utilizando la distancia coseno
    cos_sim = tf.keras.losses.cosine_similarity(feature_vec1, feature_vec2)
    return cos_sim

def get_feature_vectors(ingredient_list, max_len=64):
    # Utilizar DistilBERT para obtener los vectores de características de una lista de ingredientes
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=False)
    input_ids = tf.constant(tokenizer.encode(ingredient_list, return_tensors='tf'))
    input_ids = pad_sequences(input_ids, maxlen=max_len, padding='post')
    pooled_output = model(input_ids)[0]
    #print(pooled_output)
    feature_vec = pooled_output.numpy().squeeze()
    return feature_vec


def generar_dataset_entrenamiento(df_nutricionales='nutricion.csv',
                                  encoding='ISO-8859-1',
                                  usecols=['nombre', 'kcal', 'carbohydrate', 'protein', 'total_fat', 'sugars', 'fiber'],
                                  min_ingredientes=1,
                                  max_ingredientes=5,
                                  numero_recetas=100):
    """
    Regresa un NumPy Array para entrenar un modelo de regresión.
    Por defecto se toman las columnas: 'nombre', 'kcal','carbohydrate', 'protein', 'total_fat', 'sugars', 'fiber'
    Que son las columnas del dataframe de nutricion que usamos para entrenar.
    El método toma al azar de min_ingredientes a max_ingredientes y genera también cantidades en gramos aleatorias.
    Con esta información el método genera recetas ficticias con su correcto contenido energético y nutricional.
    Este dataset ficticio puede usarse para entrenar modelos de regresión para el cálculo energético.

    Parámetros:
    @df_nutricionales: El dataframe de donde se toman la información nutricional
    @encoding: El formato de encoding del archivo csv, por ejemplo: UTF-8 o ISO-8859-1
    @usecols: Los nombres de las columnas del csv que se codificarán en el array
    @min_ingredientes: El número mínimo de ingredientes que puede tener una receta
    @max_ingredientes: El máximo número de ingredientes que puede tener una receta

    Devuelve:
    Un NumPy Array con dtype=string (Antes de usarlo, es necesario convertir los valores numéricos a float16 o float32 etc.)

    Ejemplo:
        dataset = generar_dataset_entrenamiento(numero_recetas=1000, min_ingredientes=5, max_ingredientes=10)
    """

    df = pd.read_csv('nutricion.csv', encoding=encoding, usecols=usecols)

    print('Generando', numero_recetas, ' recetas aleatorias...\n')

    RecetaRandom = []

    for i_recetas in tqdm(range(numero_recetas)):
        nombre = ''
        kcal = 0.0
        gramos_carb = 0.0
        gramos_proteina = 0.0
        gramos_grasa = 0.0
        gramos_azucar = 0.0
        gramos_fibra = 0.0

        for i_ingredientes in range(np.random.randint(min_ingredientes, max_ingredientes + 1)):
            # Elegir un ingrediente al azar el dataframe de nutricionales
            i_rand = np.random.randint(len(df))
            cant_rand = round(np.random.ranf() * np.random.randint(1, 10), 2)
            row_alimento = df.iloc[i_rand]
            nombre += str(cant_rand) + 'gr de ' + row_alimento['nombre'].replace(',', ' ').strip() + ', '
            kcal += cant_rand * float(row_alimento['kcal'])
            gramos_carb += cant_rand * float(row_alimento['carbohydrate'].replace(' ', '').split('g')[0])
            gramos_proteina += cant_rand * float(row_alimento['protein'].replace(' ', '').split('g')[0])
            gramos_grasa += cant_rand * float(row_alimento['total_fat'].replace(' ', '').split('g')[0])
            gramos_azucar += cant_rand * float(row_alimento['sugars'].replace(' ', '').split('g')[0])
            gramos_fibra += cant_rand * float(row_alimento['fiber'].replace(' ', '').split('g')[0])

        nombre = nombre[:-2]
        RecetaRandom.append([nombre, round(kcal, 2), round(gramos_carb, 2), round(gramos_proteina, 2),
                             round(gramos_grasa, 2), round(gramos_azucar, 2), round(gramos_fibra, 2)])

    result = np.array(RecetaRandom)

    return result


def calcular_feature_vecs(array_recetas, max_len=128):
    """
    Método que recibe un array de recetas con el siguiente formato:
      - La primera columna del array debe ser un string con los ingredientes y sus cantidades:
        Formato ej: 10gr Manzana, 4.5gr azúcar, etc.
      - El resto de las columnas son valores de contenido energético (deben ser numéricos)

    La función recorre el arreglo, tokenizando y calculando el vector de características de
    la primera columna de texto, utilizando el TDistilBERTTokenizer de la función get_feature_vectors().

    Con este arreglo puede entrenarse una red neuronal que aprenda a inferir los valores energéticos
    tomando como entrada una matriz de embbedings de TDistilBERT.

    Parámetros:
    @array_recetas: Un arreglo con las recetas y su información nutricional en formato numPy array.
    @max_len: El número máximo de tokens para la matriz de embedings.

    Devuelve 2 arreglos:
      x: un arreglo con todas las matrices de embbedings para usar como entrada a un modelo
      y: un arreglo con todos los vectores de información nutricional, uno por cada matriz de embbedings.

    Ejemplo:
      dataX, dataY = calcular_feature_vecs(dataset_entrenamiento, max_len=128)
    """

    # Recorremos cada receta del array para calcular su similitud con la lista de ingredientes

    result_x = []
    result_y = []

    print('Calculando vector de características de', len(array_recetas), 'recetas...')
    for i in tqdm(range(len(array_recetas))):
        # Generando el data X con el vector caracteristicas del texto usando DistilBERT
        receta = array_recetas[i][0]
        feature_vec_receta = get_feature_vectors(receta, max_len=max_len)
        feature_vec_receta = feature_vec_receta.flatten()
        result_x.append(feature_vec_receta)

        # Generando el data Y con el resto de las columnas
        result_y.append([float(array_recetas[i][val]) for val in range(1, array_recetas.shape[1])])

    result_x = np.array(result_x, dtype=np.float16)
    result_y = np.array(result_y, dtype=np.float16)

    return result_x, result_y


# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.5)

def GenerarModelo(input_shape, numero_salidas):
    """
    Devuelve un modelo de regresión con aprendizaje profundo para aprender
    los patrones de ingredientes y sus valores nutricionales.

    Parámetros:
    @input_shape: El shape de la entrada del modelo
    @numero_salidas: El número de columnas o valores que aprenderá a predecir.

    Devuelve:
    Una instancia de la clase tensorflow.keras.Model

    Ejemplo:
      modelo = GenerarModelo(input_shape=(x_train.shape[1]), numero_salidas=y_train.shape[1])
      modelo.compile(RMSprop(learning_rate=1e-5), loss="mean_absolute_error", metrics=['accuracy'])
      modelo.summary()

      history = modelo.fit(x = x_train, y = y_train,
                           batch_size = 8,
                           epochs = 30,
                           validation_data=[x_test, y_test])

      modelo.save('MiModelo.h5')
    """

    input_tensor = Input(shape=input_shape, name='CapaEntrada')

    # Capas densamente conectadas para aprender características y patrones
    x = Dense(2048, activation='relu')(input_tensor)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    output_tensor = Dense(numero_salidas, name='CapaSalida')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor, name="ModeloRegresionNut")
    model.build(input_shape)

    return model



def EvaluarModelo(modelo, history, x_val, y_val):
    """
    Evaluar y graficar el entrenamiento de un modelo.

    Parámetros:
    @modelo: La instancia del modelo a evaluar
    @history: Una instancia del history regresado por model.fit()
    @x_val: Un array con las entradas de validación
    @y_val: Un array con los y_true de validación

    Devuelve: None
    """

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

    scores = modelo.evaluate(x_val, y_val)
    print(scores)

    test_predictions = modelo.predict(x_val)
    print(test_predictions)

    return