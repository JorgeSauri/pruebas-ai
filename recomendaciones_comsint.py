# Requerimientos de librerías:
# pip install spacy
# python -m spacy download es_core_news_md

# Importar librerías
import pandas as pd
import spacy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

class Recomendador():
    
    def __init__(self,
                 fuente='recetas.csv',
                 nutricion='nutricion.csv',
                 canasta='canasta_basica.csv',
                 encoding="ISO-8859-1"):
        """
            La clase Recomendador carga los datasets: fuente, nutricion y canasta; e inicializa los parámetros
            para recomendar una lista de recetas de acuerdo a ciertas características. Utiliza la librería de DeepLearning
            SpaCy para NLP para buscar similitudes entre el recetario (fuente) y los ingredientes de la canasta básica (canasta).

            Una vez instanciada la clase, para poder utilizarla por primera vez, hay que llamar al método: ProcesarRecetario() para
            obtener una lista de elementos más similares a la canasta básica proporcionada, y luego se llama al método Calcular_InfoNutricional()
            si se desea agregarle a ésta lista la información nutricional y costos de cada receta (Esta información es estimada, se
            utiliza SpaCy para encontrar similitudes entre cada ingrediente de la canasta básica y cada ingrediente de la receta,
            sin embargo los valores tanto nutricionales como de costos dependerán mucho del dataframe proporcionado en el parámetro 'nutricion'.

            Parámetros:
            -----------------------------------------------------------------------------------------------------------
            @fuente: Ruta y archivo csv del recetario.
            @nutricion: Ruta y archivo csv del dataset de información nutricional
            @canasta: Ruta y archivo csv de la canasta básica
            @encoding: Tipo de codificación del archivo (utf-8 o iso-8859-1)
        """


        # cargamos el modelo entrenado en español
        self.nlp = spacy.load("es_core_news_md")

        # Diccionario de medidas más comunes en recetas
        self.Medidas = {
            'miligramos': ['mg ', 'miligramo ', 'miligramos ', 'mgr ', 'mg.', 'mgr.'],
            'gramos': ['gramos ', 'gr ', 'g ', 'gram ', 'grams ', 'gr.', 'g.', 'gram.', 'grams.'],
            'onzas': ['onza ', 'onzas ', 'oz ', 'ozs ', 'onza.', 'onzas.', 'oz.', 'ozs.'],
            'kilos': ['kilo ', 'kilos ', 'kg ', 'k ', 'kgr ', 'kilo.', 'kilos.', 'kg.', 'k.', 'kgr.'],
            'mililitros': ['mililitro ', 'mililitros ', 'ml ', 'mltr ', 'mltrs ', 'ml.', 'mltr.', 'mltrs.'],
            'litros': ['litro ', 'litros ', 'l ', 'lt ', 'ltr ', 'ltrs ', 'l.', 'lt.', 'ltr.', 'ltrs.'],
            'piezas': ['pieza ', 'piezas ', 'unidad ', 'unidades ', 'pz ', 'pza ', 'pz.', 'pza.'],
            'tazas': ['taza ', 'tazas ', 'tza ', 'tz ', 'cup ', 'cups ', 'tza.', 'tz.'],
            'cucharadas': ['cucharada ', 'cucharadas ', 'cuch ', 'cda ', 'cdas ', 'cuch.', 'cda.', 'cdas.', 'tbsp ',
                           'tbsp.'],
            'cucharaditas': ['cucharadita ', 'cucharaditas ', 'cdta ', 'cdtas ', 'cdta.', 'cdtas.', 'tsp ', 'tsp.']
        }

        self.stopwords = ["el", "para", "con", "en", ",", "contra",
                     "de", "del", "la", "las", "los", "un",
                     "una", "unos", "unas", "o", "ó", "y"]

        # Dataframes:
        self.DF_RecetasFiltradas = None

        self.df_recetario = pd.read_csv(fuente, encoding=encoding)
        self.df_nutricion = pd.read_csv(nutricion, encoding=encoding)
        self.df_canasta = pd.read_csv(canasta, encoding=encoding)


    def LimpiarString(self, cadena):
        """
        Limpia una cadena de caracteres extraños, simbolos, stopwords, y unidades de medida

        @cadena: String a limpiar

        Devuelve: La cadena limpia
        """
        result = []
        for c in list(cadena.lower()):
            if (c == ';' or c == '+' or c == '-'):
                c = ','
            else:
                if (not c.isalpha()):
                    c = ' '
            result.append(c)
        result = ''.join(result).split(' ')

        result2 = []
        for e in result:
            if (e != ''):
                result2.append(e)

        result = ''
        for e in result2:
            result += str(e) + ' '


        MedidasList = []
        for medida in self.Medidas:
            for abr in self.Medidas[medida]:
                MedidasList.append(abr.strip())

        # Eliminar unidades de medida que se colaron en los ingredientes
        result = ' '.join([medida for medida in result.split(' ') if medida not in MedidasList])

        # Eliminar las stopwords comunes
        result = ' '.join([word for word in result.split(' ') if word not in self.stopwords])

        return str(result)


    def encontrar_unidades(self, cadena):
        """
        Procesa un string que presuntamente contiene la cantidad y unidad de un ingrediente
        y devuelve en que unidad se está midiendo

        @cadena: El string con la cantidad, unidad e ingrediente (Ej. '25 gramos de harina de maíz')

        Devuelve: La unidad de medida
        """

        cadena = cadena.lower()

        # Por defecto regresaremos la unidad 'pieza'
        result = 'piezas'

        for medida in self.Medidas:
            for abr in self.Medidas[medida]:
                index = cadena.find(abr)
                if index > -1:
                    result = medida
                    break

        return result


    def separar_ingredientes_spacy(self, cadena):
        """
          Recibe un string con los ingredientes mezclados y separados por coma con cantidades, unidades y descripciones,
          la procesa, y devuelve 3 listas con los ingredientes, sus cantidades y sus unidades de medida por separado

          Parámetros:
          -----------------------------------------------------------------------------------------------------------
          @cadena: String con todos los ingredientes, cantidades y unidades como viene en la receta
          -----------------------------------------------------------------------------------------------------------

          Devuelve:
          3 listas con los ingredientes separados individualmente:
          cantidades, unidades, ingredientes_texto

        """

        # Inicializa las listas para las cantidades, las unidades y los ingredientes
        cantidades = []
        unidades = []
        ingredientes_texto = []

        for cad in cadena.split(','):
            # Procesa la cadena como un documento de spaCy
            doc = self.nlp(cad)

            cantidad = 0
            unidad = None
            ingrediente_texto = ''

            # Recorre cada token en el documento
            for token in doc:
                # Si el token es un número, lo agrega a la lista de cantidades
                if token.like_num and token.text.isnumeric():
                    cantidad = float(token.text)
                    # Buscar la unidad
                    unidad = self.encontrar_unidades(cad.split(token.text)[1])
                    ingrediente_texto = cad.split(token.text)[1]
                    ingrediente_texto = self.LimpiarString(ingrediente_texto)

                    # Agrega la cantidad, la unidad y el ingrediente a las listas
                    cantidades.append(cantidad)
                    unidades.append(unidad)
                    ingredientes_texto.append(ingrediente_texto)
                    break
                else:
                    # Analizar si son fracciones en ASCII: '¼', '½', '¾'
                    # chr(188), chr(189), chr(190)
                    CharFracc = ['¼', '½', '¾', '1/4', '1/2', '1/3', '3/4']
                    NumFracc = [1 / 4, 1 / 2, 3 / 4, 1 / 4, 1 / 2, 1 / 3, 3 / 4]
                    for i in range(len(CharFracc)):
                        cf = CharFracc[i]
                        nf = NumFracc[i]
                        if token.text.strip() == cf:
                            cantidades.append(nf)
                            # Buscar unidades
                            unidad = self.encontrar_unidades(cad.split(token.text)[1])
                            unidades.append(unidad)
                            ingrediente_texto = cad.split(token.text)[1]
                            ingrediente_texto = self.LimpiarString(ingrediente_texto)
                            ingredientes_texto.append(ingrediente_texto)
                            break

                            # Devuelve las listas
        return cantidades, unidades, ingredientes_texto


    def ProcesarRecetario(self,
                          col_title='nombre_del_platillo', col_ingredientes='ingredientes',
                          similitud=0.6, max_rows=20):
        """
          Procesa el recetario cargado al instanciar la clase, y trata de encontrar las recetas más
          similares en cuanto a lista de ingredientes con el dataset de canasta básica.

          Parámetros:
          -----------------------------------------------------------------------------------------------------------
          @col_title: Nombre de la columna del csv del titulo de la receta
          @col_ingredientes: Nombre de la columna del csv con los ingredientes
          @similitud: Similitud de ingredientes mínima permitida con la lista de la canasta básica
          @max_rows: Número máximo de filas que devuelve la función ordenadas de mayor a menor similitud

          -----------------------------------------------------------------------------------------------------------

          Devuelve:
          Dataframe de pandas con las columnas: 'platillo', 'ingredientes', 'similitud'

        """


        # Limpiar el dataframe de recetas
        canasta = ','.join([prod for prod in self.df_canasta['producto']])

        # Limpiar el dataframe de información nutricional
        self.df_nutricion['nombre'] = self.df_nutricion['nombre'].str.lower()

        Platillos = []
        Ingredientes = []
        Sim = []

        print('Buscando recetas con ingredientes de la canasta básica... \n')
        for i in tqdm(range(len(self.df_recetario))):
            row = self.df_recetario.iloc[i]
            ingredientes_clean = self.LimpiarString(row[col_ingredientes])
            tokenIngredientes = self.nlp(ingredientes_clean)
            similaridad = tokenIngredientes.similarity(self.nlp(canasta))
            if similaridad > similitud:
                Platillos.append(row[col_title])
                Ingredientes.append(row[col_ingredientes])
                Sim.append(similaridad)

        dfFiltrados = pd.DataFrame(list(zip(Platillos, Ingredientes, Sim)),
                                   columns=['nombre_del_platillo', 'ingredientes', 'similitud'])

        dfFiltrados = dfFiltrados.sort_values(by=['similitud'], ascending=False)[:max_rows]

        print(' \n\n', len(dfFiltrados), 'platillos encontrados con similitud mayor a', similitud)

        # Guardamos el dataframe en una variable de la clase, y también la regresamos
        self.DF_RecetasFiltradas = dfFiltrados

        return dfFiltrados



    def Calcular_InfoNutricional(self, similitud_canasta=0.7):
        """
        Calcula la información nutricional y los costos de acuerdo al dataset
        de información nutricional y al dataset de la canasta básica

        Parámetros:
        @similitud_canasta: Porcentaje de similitud usado para evaluar un ingrediente con el dataset de la canasta básica
                            (Por defecto = 0.7 o 70%)

        Devuelve:
        El dataframe filtrado de entrada con nuevas columnas:
          kcal, proteinas_gr, carbohidratos_gr, grasas_gr, fibra_gr, azucar_gr, costo_total_min, costo_total_max
        """

        # Para poder llamar a este método, debe haberse ejecutado antes ProcesarRecetas
        if (self.DF_RecetasFiltradas == None):
            print('\nERROR: Es necesario ejecutar primero el método ProcesarRecetas()\n')
            return

        dfFiltrados = self.DF_RecetasFiltradas

        # Por cada receta:
        # 1. Extraer ingredientes individuales
        # 2. Calcular sus valores nutricionales
        # 3. Agregarlos al dataframe resultante
        print('Calculando información nutricional y costos... \n')

        Calorias = []
        Proteinas = []
        Carbs = []
        Grasas = []
        Fibras = []
        Azucares = []
        Precios_Min = []
        Precios_Max = []

        for i in tqdm(range(len(dfFiltrados))):
            # for i in range(len(dfFiltrados)):
            row = dfFiltrados.iloc[i]
            Cantidades, Unidades, Ingredientes = self.separar_ingredientes_spacy(row['ingredientes'])

            # para la receta i, recorrer sus ingredientes y cantidades y buscar su info nutricional
            kcal = 0
            gramos_proteina = 0
            gramos_carb = 0
            gramos_grasa = 0
            gramos_fibra = 0
            gramos_azucar = 0
            total_precio_min = 0
            total_precio_max = 0

            for ing_index in range(len(Ingredientes)):
                ingrediente = self.LimpiarString(Ingredientes[ing_index]).strip()
                if ingrediente == '': continue  # Se salta los vacíos

                # Buscar el ingrediente en la tabla de valores nutricionales:

                try:
                    df_ing_nut = self.df_nutricion.loc[df_nutricion['nombre'].str.contains(ingrediente.lower())][:1]
                except:  # Se coló algún caracter inválido en la búsqueda, saltarse este item
                    # print('Falla al parsear ingrediente:', ingrediente.lower())
                    EncontroAlimento = False
                    continue

                    # Si hubo coincidencias:
                EncontroAlimento = len(df_ing_nut) > 0

                if EncontroAlimento:
                    # Columnas que nos interesan:
                    # 'kcal', 'protein', 'carbohydrate', 'total_fat', 'fiber', 'sugars'
                    # Nota: Todas estas medidas son en GRAMOS
                    # Esto es tomando en cuenta que el dataset viene en la forma x.xx g:
                    # Sacamos los valores por cada 100gr (Porque así está en el dataframe)
                    kcal += int(df_ing_nut['kcal'].values[0])
                    gramos_proteina += float(df_ing_nut['protein'].values[0].replace(' ', '').split('g')[0])
                    gramos_carb += float(df_ing_nut['carbohydrate'].values[0].replace(' ', '').split('g')[0])
                    gramos_grasa += float(df_ing_nut['total_fat'].values[0].replace(' ', '').split('g')[0])
                    gramos_fibra += float(df_ing_nut['fiber'].values[0].replace(' ', '').split('g')[0])
                    gramos_azucar += float(df_ing_nut['sugars'].values[0].replace(' ', '').split('g')[0])

                    # Ahora hay que calcular cuantas porciones de 100gr tiene el ingrediente en la receta
                    # y multiplicarlo o dividirlo por 100gr:
                    cantidad = Cantidades[ing_index]
                    unidad = Unidades[ing_index]

                    # Para las unidades tipo taza o cucharada, tomaremos este parámetro:
                    # 1 taza = 200gr, 1 cucharada = 15gr, 1 cucharadita = 5gr, pieza = 150gr
                    factor = 1
                    if unidad == 'tazas': factor = 200 / 100
                    if unidad == 'cucharadas': factor = 15 / 100
                    if unidad == 'cucharadita': factor = 5 / 100
                    if unidad == 'piezas': factor = 150 / 100
                    # Medidas proporcionales al gramo
                    if unidad == 'kilos': factor = 1000 / 100
                    if unidad == 'miligramos': factor = 0.001 / 100
                    if unidad == 'onzas': factor = 28.35 / 100
                    # Medidas de líquidos
                    if unidad == 'mililitros': factor = 0.001 / 100
                    if unidad == 'litros': factor = 1000 / 100

                    kcal = kcal * (cantidad * factor)
                    gramos_proteina = gramos_proteina * (cantidad * factor)
                    gramos_carb = gramos_carb * (cantidad * factor)
                    gramos_grasa = gramos_grasa * (cantidad * factor)
                    gramos_fibra = gramos_fibra * (cantidad * factor)
                    gramos_azucar = gramos_azucar * (cantidad * factor)

                    # Buscar el costo si el ingrediente está en la canasta
                    for i_cb in range(len(self.df_canasta)):
                        row_canasta = self.df_canasta.iloc[i_cb]
                        cb_prod = row_canasta['producto']
                        cb_precio_min_gramo = 0.0
                        cb_precio_max_gramo = 0.0

                        # Si el ingrediente de la Canasta tiene una Similitud > 'similitud_canasta' con el ingrediente actual:
                        cb_ing_sim = nlp(cb_prod).similarity(nlp(ingrediente))
                        if cb_ing_sim >= similitud_canasta:
                            # print(cb_prod, 'similar a', ingrediente, 'en', cb_ing_sim)
                            cb_precio_min_gramo = float(row_canasta['precio_min_gramo'])
                            cb_precio_max_gramo = float(row_canasta['precio_max_gramo'])

                            # Multiplicamos factor * 100gr para que la unidad mínima sea por 1gr
                            total_precio_min += cantidad * (factor * 100) * cb_precio_min_gramo
                            total_precio_max += cantidad * (factor * 100) * cb_precio_max_gramo

                            break

            Calorias.append(kcal)
            Proteinas.append(gramos_proteina)
            Carbs.append(gramos_carb)
            Grasas.append(gramos_grasa)
            Fibras.append(gramos_fibra)
            Azucares.append(gramos_azucar)
            Precios_Min.append(total_precio_min)
            Precios_Max.append(total_precio_max)

        dfFiltrados['kcal'] = Calorias
        dfFiltrados['proteinas_gr'] = Proteinas
        dfFiltrados['carbohidratos_gr'] = Carbs
        dfFiltrados['grasas_gr'] = Grasas
        dfFiltrados['fibra_gr'] = Fibras
        dfFiltrados['azucar_gr'] = Azucares
        dfFiltrados['costo_total_min'] = Precios_Min
        dfFiltrados['costo_total_max'] = Precios_Max

        self.DF_RecetasFiltradas = dfFiltrados

        return dfFiltrados

