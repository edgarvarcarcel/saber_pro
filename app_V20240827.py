# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 23:11:01 2024

"""
###############################################################################
# Cargar librerias

# App
import streamlit as st
from streamlit_option_menu import option_menu

# Utilidades
import pandas as pd
import pickle as pkl
import numpy as np
from itertools import product
import joblib
import pandas as pd, numpy as np
import pickle
import shap


print('Librerias cargadas')

###############################################################################
# Inicialización de la aplicación
@st.cache_data() 
def initialize_app():
    # Cargar datos de popayan
    datos_popayan_ruta = r'data'
    datos_popayan_nombre = r'/Datos_Popayan.xlsx'
    datos_popayan = pd.read_excel(datos_popayan_ruta + datos_popayan_nombre)
    datos_popayan = datos_popayan.dropna().reset_index(drop = True)
    print('Archivo cargado -->' , datos_popayan_ruta + datos_popayan_nombre)
    
    # Cargar modelo
    path_model = r'models'
    model_filename = r'/Pipeline_V20240822.pkl'
    with open(path_model + model_filename, 'rb') as file:
        model = joblib.load(file)
    print('Archivo cargado -->' , path_model + model_filename)
    
    # Cargar shap values
    path_shap= r'models'
    shap_filename = r'/shap_values.pkl'
    with open(path_shap + shap_filename, 'rb') as f:
        data = pickle.load(f)
        loaded_shap_values = data['shap_values']
        loaded_base_values = data['base_values']
        
    # Cargar archivo de parámetros
    codificador_ordinales = pd.read_excel(r'data/Parametros.xlsx' ,
                                      sheet_name = 'Codificacion_Ordinales',
                                      dtype = {'COLUMNA' : str , 
                                                'VALOR_ACTUAL' : str})

    # Cargar ordinal encoder
    with open(r'models/Categorical_Encoder.sav' , 'rb') as file:
        cat_encoder = pickle.load(file)
    
    # Procesar la base de popayán para los ejemplos
    target = ['PUNTAJE_GLOBAL'] +  ['MOD_RAZONA_CUANTITAT_PUNT',
                                 'MOD_COMUNI_ESCRITA_PUNT',
                                 'MOD_INGLES_PUNT',
                                 'MOD_LECTURA_CRITICA_PUNT',
                                 'MOD_COMPETEN_CIUDADA_PUNT']
    ordinal_features = ['ESTU_VALORMATRICULAUNIVERSIDAD' , 'ESTU_HORASSEMANATRABAJA' , 'FAMI_ESTRATOVIVIENDA',
                   'FAMI_EDUCACIONPADRE' , 'FAMI_EDUCACIONMADRE']
    
    numeric_features = []
    categorical_yes_encoding = cat_encoder.feature_names_in_.tolist()
    categorical_no_encoding = ['ESTU_COD_RESIDE_DEPTO',
                             'ESTU_COD_DEPTO_PRESENTACION',
                             'INST_COD_INSTITUCION']
    
    # Filtrar los nombres de las características que no fueron eliminadas por VarianceThreshold
    selected_features = ['INST_CARACTER_ACADEMICO', 'ESTU_METODO_PRGM', 'ESTU_PAGOMATRICULABECA',
                         'ESTU_PAGOMATRICULACREDITO', 'ESTU_GENERO', 'ESTU_PAGOMATRICULAPADRES',
                         'ESTU_PAGOMATRICULAPROPIO', 'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENELAVADORA',
                         'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'INST_ORIGEN', 'ESTU_VALORMATRICULAUNIVERSIDAD_ENCODED',
                         'ESTU_HORASSEMANATRABAJA_ENCODED', 'FAMI_ESTRATOVIVIENDA_ENCODED', 'FAMI_EDUCACIONPADRE_ENCODED',
                         'FAMI_EDUCACIONMADRE_ENCODED', 'ESTU_COD_RESIDE_DEPTO', 'ESTU_COD_DEPTO_PRESENTACION', 'INST_COD_INSTITUCION']

    
    # Aplicar codificacion de variables
    for i in ordinal_features:
        aux = codificador_ordinales[codificador_ordinales['COLUMNA'] == i]
        datos_popayan[i + '_ENCODED'] = datos_popayan[i].map(dict(aux[['VALOR_ACTUAL' , 'VALOR_CODIFICADO']].values))
    # Actualizar las variables ordinales
    ordinal_features = [i for i in datos_popayan.columns.tolist() if '_ENCODED' in i]
    # Aplicar el Ordinal Encoder

    aux_categorical = pd.DataFrame(cat_encoder.transform(datos_popayan[categorical_yes_encoding]) , columns = categorical_yes_encoding)
    
    # Concatenar la información
    data_3 = pd.concat([aux_categorical.reset_index(drop = True),
                        datos_popayan[ordinal_features].reset_index(drop = True),
                        datos_popayan[categorical_no_encoding].reset_index(drop = True),
                        datos_popayan[target].reset_index(drop = True)] , axis = 1)
    # Definir los tipos de datos para las variables
    for i in data_3.columns.tolist():
        if i in numeric_features or i in target:
            data_3[i] = data_3[i].fillna(-1).astype(float)
        if i in ordinal_features or i in categorical_no_encoding or i in categorical_yes_encoding:
            data_3[i] = data_3[i].fillna(-1).astype(int)
    
    X_popayan = data_3.drop(columns = target)
    
    # Transformar los datos con el pipeline       
    X_transformed = model.best_estimator_[:-1].transform(X_popayan)
    
    # Obtener el explainer
    explainer = shap.KernelExplainer(model.best_estimator_.named_steps['catboostregressor'].predict, X_transformed)
    
    print('App inicializada')
    
    return datos_popayan ,  codificador_ordinales , cat_encoder ,  target , ordinal_features , categorical_yes_encoding , categorical_no_encoding , selected_features , model , X_popayan , X_transformed , explainer

# Funcion to process user input
def parser_user_input(dataframe_input , codificador_ordinales , cat_encoder ,  target , ordinal_features , categorical_yes_encoding , categorical_no_encoding , selected_features , diccionario_variables , explainer):

    # Aplicar la categorización de las variables
    ordinal_features = ['ESTU_VALORMATRICULAUNIVERSIDAD' , 'ESTU_HORASSEMANATRABAJA' , 'FAMI_ESTRATOVIVIENDA',
                   'FAMI_EDUCACIONPADRE' , 'FAMI_EDUCACIONMADRE']
    
    # Aplicar codificacion de variables
    for i in ordinal_features:
        aux = codificador_ordinales[codificador_ordinales['COLUMNA'] == i]
        dataframe_input[i + '_ENCODED'] = dataframe_input[i].map(dict(aux[['VALOR_ACTUAL' , 'VALOR_CODIFICADO']].values))
    # Actualizar las variables ordinales
    ordinal_features = [i for i in datos_popayan.columns.tolist() if '_ENCODED' in i]
    # Aplicar el Ordinal Encoder

    aux_categorical = pd.DataFrame(cat_encoder.transform(dataframe_input[categorical_yes_encoding]) , columns = categorical_yes_encoding)
    
    # Concatenar la información
    data_3 = pd.concat([aux_categorical.reset_index(drop = True),
                        dataframe_input[ordinal_features].reset_index(drop = True),
                        dataframe_input[categorical_no_encoding].reset_index(drop = True)] ,
                       axis = 1)
    
    # Transformar las variables categóricas faltantes
    for i in diccionario_variables.keys():
        data_3[i] = data_3[i].map(diccionario_variables[i])
    
    # Definir los tipos de datos para las variables
    for i in data_3.columns.tolist():
        if i in ordinal_features or i in categorical_no_encoding or i in categorical_yes_encoding:
            data_3[i] = data_3[i].fillna(-1).astype(int)
    
    # Organizar columnas exactamente en el mismo orden para el modelo
    data_3 = data_3[model.feature_names_in_.tolist()]
    
    # Realizar prediccion del puntaje global
    prediccion = model.predict(data_3)[0]
    
    # Calcular los valores shap
    X_transformed = model.best_estimator_[:-1].transform(data_3)
    shap_values = explainer(X_transformed)
    
    # Colocar los labels originales
    explicacion = shap.Explanation(shap_values.values, 
                      shap_values.base_values, 
                      data = X_transformed, 
                      feature_names = selected_features)
    
    # Crear el waterfall plot
    plot = shap.plots.waterfall(explicacion[0])
    
    # Mostrar la figura
    st.pyplot(plot)
    
    # Mostrar mensaje con la predicción
    mensaje = f"Con la información proporcionada, se estima que el puntaje promedio en los módulos del estudiante será de **{round(prediccion , 2)}** puntos"
    st.write(mensaje)
    return None


###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize app
datos_popayan ,  codificador_ordinales , cat_encoder ,  target , ordinal_features , categorical_yes_encoding , categorical_no_encoding , selected_features , model , X_popayan , X_transformed , explainer = initialize_app()

# Generar un diccionario para aquellas variables que no se codifican por el mismo modelo
aux_estu_cod_reside_depto = {datos_popayan[datos_popayan['ESTU_COD_RESIDE_DEPTO'] == i]['ESTU_DEPTO_RESIDE'].values[0] : i for i in  datos_popayan['ESTU_COD_RESIDE_DEPTO'].unique().tolist()}
aux_estu_cod_reside_depto_2 = {i : datos_popayan[datos_popayan['ESTU_COD_RESIDE_DEPTO'] == i]['ESTU_DEPTO_RESIDE'].values[0]  for i in  datos_popayan['ESTU_COD_RESIDE_DEPTO'].unique().tolist() }
aux_estu_cod_depto_presentacion = {aux_estu_cod_reside_depto_2[i] : i for i in datos_popayan['ESTU_COD_DEPTO_PRESENTACION'].unique().tolist() if i in aux_estu_cod_reside_depto_2.keys()}
aux_estu_cod_depto_presentacion['BOGOTÁ D.C'] = 11
aux_inst_cod_institucion = {datos_popayan[datos_popayan['INST_COD_INSTITUCION'] == i]['INST_NOMBRE_INSTITUCION'].values[0] : i for i in datos_popayan['INST_COD_INSTITUCION'].unique().tolist()}


diccionario_variables = {'ESTU_COD_RESIDE_DEPTO' : aux_estu_cod_reside_depto ,
                         'ESTU_COD_DEPTO_PRESENTACION' : aux_estu_cod_depto_presentacion,
                         'INST_COD_INSTITUCION' : aux_inst_cod_institucion}


# Configuración de opciones del menú
with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Principal' , 'Predicción' , 'Ejemplos'],
        icons = ['house' , 'book' , 'book'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'Vertical')
    
######################
# Layout de la página principal
######################
if selected == 'Principal':
    st.title('Análisis de Factores')
    st.markdown("""
    Esta aplicación contiene tres secciones las cuales pueden ser accedidas mediante el menú horizontal.\n
    Las secciones son:\n
    **Principal**: Página de bienvenida.\n
    **Predicción:** En esta sección puede seleccionar las características socio-demográficas
    del estudiante y se realizará la predicción de su puntaje global, así como el análisis de factores.\n
    **Ejemplos:** En esta sección puede seleccionar ejemplos pre-cargados.
    """)
    
###############################################################################
# Layout página predicción
if selected == 'Predicción':
    st.title('Sección de precicción')
    st.subheader("Descripción")
    st.subheader("Pare realizar el proceso, siga los siguientes pasos:")
    st.markdown("""
    1. Ingrese la información del estudiante en las opciones de la barra de la izquierda. En caso que no aplique, seleccione la opción nan
    2. Presione el botón "Realizar predicción" y espere por los resultados.
    """)
    st.markdown("""
    Este modelo predice el posible puntaje global que obtendrá el estudiante, así como los factores que inciden en dicha predicción
    """)
    # Layout de la información del estudiante
    st.sidebar.title("Información del estudiante")
    st.sidebar.subheader("Seleccione los parámetros")
    
    # Variables de entrada
    
    INST_CARACTER_ACADEMICO = st.sidebar.selectbox('Carácter académico de la institución',
                                                   tuple(cat_encoder.categories_[0].tolist()))
    
    ESTU_METODO_PRGM = st.sidebar.selectbox('Tipo de programa académico',
                                                   tuple(cat_encoder.categories_[1].tolist()))
    
    ESTU_PAGOMATRICULABECA = st.sidebar.selectbox('¿Estudiante pagó matricula?',
                                                   tuple(cat_encoder.categories_[2].tolist()))
    
    ESTU_PAGOMATRICULACREDITO = st.sidebar.selectbox('¿Estudiante pagó matricula a crédito?',
                                                   tuple(cat_encoder.categories_[3].tolist()))
    
    ESTU_GENERO = st.sidebar.selectbox('Género del estudiante',
                                                   tuple(cat_encoder.categories_[4].tolist()))
    
    ESTU_PAGOMATRICULAPADRES = st.sidebar.selectbox('¿La matricula la pagaron los padres del estudiante?',
                                                   tuple(cat_encoder.categories_[5].tolist()))
    
    ESTU_ESTADOINVESTIGACION = st.sidebar.selectbox('¿Cuál es el estado de la publicación del estudiante, si aplica?',
                                                   tuple(cat_encoder.categories_[6].tolist()))
    
    ESTU_PAGOMATRICULAPROPIO = st.sidebar.selectbox('¿El estudiante pagó la matricula por sí mismo?',
                                                   tuple(cat_encoder.categories_[7].tolist()))
    
    FAMI_TIENEAUTOMOVIL = st.sidebar.selectbox('¿La familia del estudiante tiene automovil?',
                                                   tuple(cat_encoder.categories_[8].tolist()))
    
    FAMI_TIENELAVADORA = st.sidebar.selectbox('¿La familia del estudiante tiene lavadora?',
                                                   tuple(cat_encoder.categories_[9].tolist()))
    
    FAMI_TIENECOMPUTADOR = st.sidebar.selectbox('¿La familia del estudiante tiene computador?',
                                                   tuple(cat_encoder.categories_[10].tolist()))
    
    FAMI_TIENEINTERNET = st.sidebar.selectbox('¿La familia del estudiante tiene internet?',
                                                   tuple(cat_encoder.categories_[11].tolist()))
    
    INST_ORIGEN = st.sidebar.selectbox('¿Cuál es el tipo de institución donde estudió?',
                                                   tuple(cat_encoder.categories_[12].tolist()))
    
    ESTU_HORASSEMANATRABAJA = st.sidebar.selectbox('¿Cuántas horas a la semana trabaja el estudiante?',
                                                   tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'ESTU_HORASSEMANATRABAJA']['VALOR_ACTUAL'].values.tolist()))
    
    ESTU_VALORMATRICULAUNIVERSIDAD = st.sidebar.selectbox('¿Cuál fue el valor pagado de la matrícula en la universidad?',
                                                   tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'ESTU_VALORMATRICULAUNIVERSIDAD']['VALOR_ACTUAL'].values.tolist()))

    FAMI_EDUCACIONMADRE = st.sidebar.selectbox('¿Cuál es el nivel académico de la madre?',
                                                   tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'FAMI_EDUCACIONMADRE']['VALOR_ACTUAL'].values.tolist()))

    FAMI_EDUCACIONPADRE = st.sidebar.selectbox('¿Cuál es el nivel académico del padre?',
                                                    tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'FAMI_EDUCACIONPADRE']['VALOR_ACTUAL'].values.tolist()))

    FAMI_ESTRATOVIVIENDA = st.sidebar.selectbox('¿Cuál es el estrato socio económico de la familia?',
                                                    tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'FAMI_ESTRATOVIVIENDA']['VALOR_ACTUAL'].values.tolist()))

    ESTU_COD_RESIDE_DEPTO = st.sidebar.selectbox('¿Cuál es el departamento donde reside el estudiante?',
                                                 tuple(aux_estu_cod_reside_depto.keys()))
    
    ESTU_COD_DEPTO_PRESENTACION = st.sidebar.selectbox('¿Cuál es el departamento donde el estudiante presentó la prueba saber pro?',
                                                 tuple(aux_estu_cod_depto_presentacion.keys()))
    
    INST_COD_INSTITUCION = st.sidebar.selectbox('¿Con qué insitución se está presentando en la prueba saber pro?',
                                                 tuple(aux_inst_cod_institucion.keys()))

   
    dataframe_input = pd.DataFrame({'INST_CARACTER_ACADEMICO' : [INST_CARACTER_ACADEMICO],
                                    'ESTU_METODO_PRGM' : [ESTU_METODO_PRGM],
                                    'ESTU_PAGOMATRICULABECA' : [ESTU_PAGOMATRICULABECA],
                                    'ESTU_PAGOMATRICULACREDITO' : [ESTU_PAGOMATRICULACREDITO],
                                    'ESTU_GENERO' : [ESTU_GENERO],
                                    'ESTU_PAGOMATRICULAPADRES' : [ESTU_PAGOMATRICULAPADRES],
                                    'ESTU_ESTADOINVESTIGACION' : [ESTU_ESTADOINVESTIGACION],
                                    'ESTU_PAGOMATRICULAPROPIO' : [ESTU_PAGOMATRICULAPROPIO],
                                    'FAMI_TIENEAUTOMOVIL' : [FAMI_TIENEAUTOMOVIL],
                                    'FAMI_TIENELAVADORA' : [FAMI_TIENELAVADORA],
                                    'FAMI_TIENECOMPUTADOR' : [FAMI_TIENECOMPUTADOR],
                                    'FAMI_TIENEINTERNET' : [FAMI_TIENEINTERNET],
                                    'INST_ORIGEN' : [INST_ORIGEN],
                                    'ESTU_HORASSEMANATRABAJA' : [ESTU_HORASSEMANATRABAJA],
                                    'ESTU_VALORMATRICULAUNIVERSIDAD' : [ESTU_VALORMATRICULAUNIVERSIDAD],
                                    'FAMI_EDUCACIONMADRE' : [FAMI_EDUCACIONMADRE],
                                    'FAMI_EDUCACIONPADRE' : [FAMI_EDUCACIONPADRE],
                                    'FAMI_ESTRATOVIVIENDA' : [FAMI_ESTRATOVIVIENDA],
                                    'ESTU_COD_RESIDE_DEPTO' : [ESTU_COD_RESIDE_DEPTO],
                                    'ESTU_COD_DEPTO_PRESENTACION' : [ESTU_COD_DEPTO_PRESENTACION],
                                    'INST_COD_INSTITUCION' : [INST_COD_INSTITUCION]})
    # Parser input and make predictions
    predict_button = st.button('Realizar predicción')
    if predict_button:
        predictions = parser_user_input(dataframe_input , codificador_ordinales , cat_encoder ,  target , ordinal_features , categorical_yes_encoding , categorical_no_encoding , selected_features , diccionario_variables , explainer)
        
##############################################################################
# Example page layout
# Prediction page layout
if selected == 'Ejemplos':
    
    st.title('Ejemplos')
    st.subheader("Descripción")
    st.subheader("Esta sección tiene diferentes ejemplos de estudiantes pre-cargados, siga las instrucciones:")
    st.markdown("""
    1. Seleccione el número del ejemplo que desea utilizar.
    2. Presione el botón "Realizar predicción" y espere los resultados.
    """)
    st.markdown("""
    Este modelo predice el posible puntaje global que obtendrá el estudiante, así como los factores que inciden en dicha predicción.
    """)
    
    fila_estudiante = st.slider("Númeor del ejemplo a utilizar:", min_value = 0, max_value = X_popayan.shape[0] - 1,step = 1)
    aux_x_popayan = pd.DataFrame(X_popayan.loc[fila_estudiante , :]).T
    aux_x_transformed = pd.DataFrame(X_transformed[fila_estudiante , :]).T
    aux_datos_popayan = pd.DataFrame(datos_popayan.loc[fila_estudiante , :]).T
    # Sidebar layout
    st.sidebar.title("Información del estudiante")
    st.sidebar.subheader("Los parámetros se fijan automáticamente con base en el ejemplo seleccionado")
    
    # Variables de entrada
    INST_CARACTER_ACADEMICO = st.sidebar.selectbox('Carácter académico de la institución',
                                                   tuple(cat_encoder.categories_[0].tolist()),
                                                   index = tuple(cat_encoder.categories_[0].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[0]].values[0]))
    
    ESTU_METODO_PRGM = st.sidebar.selectbox('Tipo de programa académico',
                                                   tuple(cat_encoder.categories_[1].tolist()),
                                                   index = tuple(cat_encoder.categories_[1].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[1]].values[0]))
    
    ESTU_PAGOMATRICULABECA = st.sidebar.selectbox('¿Estudiante pagó matricula?',
                                                   tuple(cat_encoder.categories_[2].tolist()),
                                                   index = tuple(cat_encoder.categories_[2].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[2]].values[0]))
    
    ESTU_PAGOMATRICULACREDITO = st.sidebar.selectbox('¿Estudiante pagó matricula a crédito?',
                                                   tuple(cat_encoder.categories_[3].tolist()),
                                                   index = tuple(cat_encoder.categories_[3].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[3]].values[0]))
    
    ESTU_GENERO = st.sidebar.selectbox('Género del estudiante',
                                                   tuple(cat_encoder.categories_[4].tolist()),
                                                   index = tuple(cat_encoder.categories_[4].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[4]].values[0]))
    
    ESTU_PAGOMATRICULAPADRES = st.sidebar.selectbox('¿La matricula la pagaron los padres del estudiante?',
                                                   tuple(cat_encoder.categories_[5].tolist()),
                                                   index = tuple(cat_encoder.categories_[5].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[5]].values[0]))
    
    ESTU_ESTADOINVESTIGACION = st.sidebar.selectbox('¿Cuál es el estado de la publicación del estudiante, si aplica?',
                                                   tuple(cat_encoder.categories_[6].tolist()),
                                                   index = tuple(cat_encoder.categories_[6].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[6]].values[0]))
    
    ESTU_PAGOMATRICULAPROPIO = st.sidebar.selectbox('¿El estudiante pagó la matricula por sí mismo?',
                                                   tuple(cat_encoder.categories_[7].tolist()),
                                                   index = tuple(cat_encoder.categories_[7].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[7]].values[0]))
    
    FAMI_TIENEAUTOMOVIL = st.sidebar.selectbox('¿La familia del estudiante tiene automovil?',
                                                   tuple(cat_encoder.categories_[8].tolist()),
                                                   index = tuple(cat_encoder.categories_[8].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[8]].values[0]))
    
    FAMI_TIENELAVADORA = st.sidebar.selectbox('¿La familia del estudiante tiene lavadora?',
                                                   tuple(cat_encoder.categories_[9].tolist()),
                                                   index = tuple(cat_encoder.categories_[9].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[9]].values[0]))
    
    FAMI_TIENECOMPUTADOR = st.sidebar.selectbox('¿La familia del estudiante tiene computador?',
                                                   tuple(cat_encoder.categories_[10].tolist()),
                                                   index = tuple(cat_encoder.categories_[10].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[10]].values[0]))
    
    FAMI_TIENEINTERNET = st.sidebar.selectbox('¿La familia del estudiante tiene internet?',
                                                   tuple(cat_encoder.categories_[11].tolist()),
                                                   index = tuple(cat_encoder.categories_[11].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[11]].values[0]))
    
    INST_ORIGEN = st.sidebar.selectbox('¿Cuál es el tipo de institución donde estudió?',
                                                   tuple(cat_encoder.categories_[12].tolist()),
                                                   index = tuple(cat_encoder.categories_[12].tolist()).index(aux_datos_popayan[cat_encoder.feature_names_in_.tolist()[12]].values[0]))
    
    ESTU_HORASSEMANATRABAJA = st.sidebar.selectbox('¿Cuántas horas a la semana trabaja el estudiante?',
                                                   tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'ESTU_HORASSEMANATRABAJA']['VALOR_ACTUAL'].values.tolist()),
                                                   index = tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'ESTU_HORASSEMANATRABAJA']['VALOR_ACTUAL'].values.tolist()).index(aux_datos_popayan['ESTU_HORASSEMANATRABAJA'].values[0]))
    
    ESTU_VALORMATRICULAUNIVERSIDAD = st.sidebar.selectbox('¿Cuál fue el valor pagado de la matrícula en la universidad?',
                                                   tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'ESTU_VALORMATRICULAUNIVERSIDAD']['VALOR_ACTUAL'].values.tolist()),
                                                   index = tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'ESTU_VALORMATRICULAUNIVERSIDAD']['VALOR_ACTUAL'].values.tolist()).index(aux_datos_popayan['ESTU_VALORMATRICULAUNIVERSIDAD'].values[0]))

    FAMI_EDUCACIONMADRE = st.sidebar.selectbox('¿Cuál es el nivel académico de la madre?',
                                                   tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'FAMI_EDUCACIONMADRE']['VALOR_ACTUAL'].values.tolist()),
                                                   index = tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'FAMI_EDUCACIONMADRE']['VALOR_ACTUAL'].values.tolist()).index(aux_datos_popayan['FAMI_EDUCACIONMADRE'].values[0]))

    FAMI_EDUCACIONPADRE = st.sidebar.selectbox('¿Cuál es el nivel académico del padre?',
                                                    tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'FAMI_EDUCACIONPADRE']['VALOR_ACTUAL'].values.tolist()),
                                                    index = tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'FAMI_EDUCACIONPADRE']['VALOR_ACTUAL'].values.tolist()).index(aux_datos_popayan['FAMI_EDUCACIONPADRE'].values[0]))

    FAMI_ESTRATOVIVIENDA = st.sidebar.selectbox('¿Cuál es el estrato socio económico de la familia?',
                                                    tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'FAMI_ESTRATOVIVIENDA']['VALOR_ACTUAL'].values.tolist()),
                                                    index = tuple(codificador_ordinales[codificador_ordinales['COLUMNA'] == 'FAMI_ESTRATOVIVIENDA']['VALOR_ACTUAL'].values.tolist()).index(aux_datos_popayan['FAMI_ESTRATOVIVIENDA'].values[0]))

    ESTU_COD_RESIDE_DEPTO = st.sidebar.selectbox('¿Cuál es el departamento donde reside el estudiante?',
                                                 tuple(aux_estu_cod_reside_depto.keys()),
                                                 index = tuple(aux_estu_cod_reside_depto.keys()).index(aux_datos_popayan['ESTU_DEPTO_RESIDE'].values[0]))
    
    ESTU_COD_DEPTO_PRESENTACION = st.sidebar.selectbox('¿Cuál es el departamento donde el estudiante presentó la prueba saber pro?',
                                                 tuple(aux_estu_cod_depto_presentacion.keys()),
                                                 index = tuple(aux_estu_cod_depto_presentacion.values()).index(aux_datos_popayan['ESTU_COD_DEPTO_PRESENTACION'].values[0]))
    
    INST_COD_INSTITUCION = st.sidebar.selectbox('¿Con qué insitución se está presentando en la prueba saber pro?',
                                                 tuple(aux_inst_cod_institucion.keys()),
                                                 index = tuple(aux_inst_cod_institucion.values()).index(aux_datos_popayan['INST_COD_INSTITUCION'].values[0]))
    
    
    dataframe_input = pd.DataFrame({'INST_CARACTER_ACADEMICO' : [INST_CARACTER_ACADEMICO],
                                    'ESTU_METODO_PRGM' : [ESTU_METODO_PRGM],
                                    'ESTU_PAGOMATRICULABECA' : [ESTU_PAGOMATRICULABECA],
                                    'ESTU_PAGOMATRICULACREDITO' : [ESTU_PAGOMATRICULACREDITO],
                                    'ESTU_GENERO' : [ESTU_GENERO],
                                    'ESTU_PAGOMATRICULAPADRES' : [ESTU_PAGOMATRICULAPADRES],
                                    'ESTU_ESTADOINVESTIGACION' : [ESTU_ESTADOINVESTIGACION],
                                    'ESTU_PAGOMATRICULAPROPIO' : [ESTU_PAGOMATRICULAPROPIO],
                                    'FAMI_TIENEAUTOMOVIL' : [FAMI_TIENEAUTOMOVIL],
                                    'FAMI_TIENELAVADORA' : [FAMI_TIENELAVADORA],
                                    'FAMI_TIENECOMPUTADOR' : [FAMI_TIENECOMPUTADOR],
                                    'FAMI_TIENEINTERNET' : [FAMI_TIENEINTERNET],
                                    'INST_ORIGEN' : [INST_ORIGEN],
                                    'ESTU_HORASSEMANATRABAJA' : [ESTU_HORASSEMANATRABAJA],
                                    'ESTU_VALORMATRICULAUNIVERSIDAD' : [ESTU_VALORMATRICULAUNIVERSIDAD],
                                    'FAMI_EDUCACIONMADRE' : [FAMI_EDUCACIONMADRE],
                                    'FAMI_EDUCACIONPADRE' : [FAMI_EDUCACIONPADRE],
                                    'FAMI_ESTRATOVIVIENDA' : [FAMI_ESTRATOVIVIENDA],
                                    'ESTU_COD_RESIDE_DEPTO' : [ESTU_COD_RESIDE_DEPTO],
                                    'ESTU_COD_DEPTO_PRESENTACION' : [ESTU_COD_DEPTO_PRESENTACION],
                                    'INST_COD_INSTITUCION' : [INST_COD_INSTITUCION]})
    # Parser input and make predictions
    predict_button = st.button('Realizar predicción')
    if predict_button:
        predictions = parser_user_input(dataframe_input , codificador_ordinales , cat_encoder ,  target , ordinal_features , categorical_yes_encoding , categorical_no_encoding , selected_features , diccionario_variables , explainer)
        