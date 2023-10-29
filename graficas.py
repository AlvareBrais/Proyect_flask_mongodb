# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:20:16 2023

@author: Brais
"""

"""
Realice un proyecto en flask y mongodb para crear una app web para visualizar los resultados
de un torneo de videojuegos.

A contuniacion realizare un analisis de datos, sobre el archivo json con los jugadores
que participaron en el torneo(se generaron de manera aleatorio con ciertos rangos)

El codigo lo he realizado mientras realizaba el curso "tu primera semana como data science",
de la empresa: data science 4 business

Los 3 componentes a desarrollar en nuestro proyecto son:
    1º) Business Anlytics
    2º) Machine learning
    3º) Productivización
"""

#Importamos las librerias necesarias.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

#Carga de los datos.

#Creamos el df que con pandas es tan facil como:
df = pd.read_json('players.json')

# Ahora, el DataFrame 'df' contiene los datos del archivo JSON


#EDA Exploratory Data Analysis

#Visualizamos las primeras 10 filas.
print()
print(df.head(10))



print()

column_names = df.columns

# Imprime los nombres de las columnas
print("Los nombres de las columnas son:" ,column_names)
print()
print("Ahora vemos de que es cada tipo de columna para hacernos una idea general")

#Visualizamos de que es cada tipo de dato por columna
print()
print(df.info()) #importante los parantesis despues del info, si no muestra todo el documento

#Comprobacion de la calidad de datos
#Realizamos un analisis de nulos y buscamos datos atipicos
nulos = df.isnull() #Creamos un df igual al df original pero con valos false si no son nulos 
                    # y True si son nulos.( en este caso no hay nulos.)
#print("Cantidad de nulos -->",nulos) imprime una matriz con todo false, ya que no hay nulos
      
nulos_por_columna = nulos.sum()

#Mostramos la cantidad de nulos por columna
print("Cantidad de nulos por columna:")
print(nulos_por_columna)



#Eda variables categoricas
#Intentar encontrar algun problema con los datos que se nos haya escapado.

#Var categoricas -> No son numericas, boleanos por ejemplo

#Var numericas -> Numeros

#Creamos una funcion para crear graficos de las var categoricas
#Tendriamos que eliminar el campo nombre
def grafica_categoricas(cat):
     #Calculamos el n de filas que necesitamos
     
    #Eliminamos la columna nombre
    if 'nombres' in df.columns:
         df.drop('nombres', axis=1)
         
    from math import ceil
    filas = ceil(cat.shape[1]/2)
    
    #Definimos el grafico
    f, ax = plt.subplots(nrows = filas, ncols= 2, figsize = (16,filas * 6))
    
    #Aplanamos para iterer por el grafico como si fuera de 1 dimension en lugar 2
    ax = ax.flat
    
    #Creamos buble que va añadiendo los diferentes graficos
    
    for cada, variable in enumerate(cat):
        cat[variable].value_counts().plot.barh(ax = ax[cada])
        ax[cada].set_title(variable,fontsize = 12,fontweight = "bold")
        ax[cada].tick_params(labelsize=12)


grafica_categoricas(df.select_dtypes("object"))
"""
Analizando la grafica vemos como la posicion mas usada es Mid, seguido de jungla y top, como pasaria en un
torneo real, ya que mid es la posicion mas demandada en partidas online.
Respecto a las posiciones, vemos que platino y diamante son las mas comunes, tiene sentido
ya que al ser un torneo los jugadores que se presentan son en su mayoria jugadores 
muy experimentados, los cuales solo tienen el 16% de todos los jugadores por encima de ellos, 
en el caso de diamante baja hasta un 6.
Observamos como maestro top 0.38% es la que menos tiene por debajo de hierro o oro.
"""


#EDA para variables numericas.
#Creamos una funcion para poder analizar de manera ESCALABLE, todas las variables numericas del dataset
#Al ser numerico en vez de hacer analisis grafico, realizamos uno estadistico
"""
def estadistico_cont(num):
    #Calculamos describe
    estadisticos = num.describe().T
    #Añadimos la mediana
    estadisticos["median"]= num.median()
    
    #Reordenamos para que la mediana este al lado de la media
    estadisticos = estadisticos.loc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'median']]
    #Lo devolvemos
    return(estadisticos)

resultados = estadistico_cont(df.select_dtypes("number"))
"""

print()

print("EDA de las var numericas: ")

print()
def estadistico_cont(num):
    # Calculamos describe
    estadisticos = num.describe().T
    
    # Añadimos la mediana
    estadisticos["median"] = num.median()
    
    # Reordenamos para que la mediana esté al lado de la media
    estadisticos = estadisticos[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'median']]
    
    # Reiniciamos el índice para que esté en el orden deseado
    estadisticos.reset_index(inplace=True)
    
    # Cambiamos el nombre de la columna del índice a 'variable'
    estadisticos.rename(columns={'index': 'variable'}, inplace=True)
    
    # Lo devolvemos
    return estadisticos


# Configura la opción para mostrar todas las columnas
pd.set_option('display.max_columns', None)
# Llama a la función y pasa el DataFrame con las columnas numéricas
resultados = estadistico_cont(df.select_dtypes("number"))

print(resultados)

#Conclusiones
#La diferencia entre la media y la mediana nos indica que hay ciertos jugadores que han jugado un numero mucho mayor
#y menor de partidas, la presencia de estos datos atipicos puede interferir en la dispersion de los datos.

#Realizamos un analisis mas detallado 
#Analisis de distribucion de fre4cuencia y grafico de caja(box plot)para intentar endtender mejor la distribucion de lso datos.
# Crear un gráfico de caja (box plot) para el número de partidas
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['npartidas'])
plt.title('Distribución del Número de Partidas Jugadas')
plt.xlabel('Número de Partidas')

plt.show()

"""
Realizamos unas pregunta semilla que intentaremos contestar

"El mayor elo es el que tiene mas % de victorias?"

"""

# Ordenamos el DataFrame por la columna 'winrate' en orden descendente
top_jugadores = df.sort_values(by='winrate', ascending=False).head(3)

# Imprime los 3 primeros jugadores con el winrate más alto
print(top_jugadores)

# Crea una gráfica de barras para los 3 jugadores con el winrate más alto
plt.figure(figsize=(8, 6))
plt.bar(top_jugadores['elo'], top_jugadores['winrate'])
plt.title('Top 3 Jugadores con Mayor Winrate')
plt.xlabel('Jugador')
plt.ylabel('Winrate')
plt.xticks(rotation=45)

# Muestra la gráfica
plt.show()



#"El mayor elo es el que tiene mas % de victorias?"

#Haciendo una grafica del top 3 vemos que 2 tienen los elos mas altos y unos de los mas bajos,
#lo que nos puede indicar que este jugador de elo bajo esta haciendo trampas, y ese no es su elo real.
#Ese jugador bronze lo podemos considerar outlier

"""
MACHINE LEARNING, tiene varios pasos

1º) preparar los datos para la modelizacion
2º) preparar las variables categoricas a numericas
3º) diseño de la modelizacion
4º) Entrenamiento del modelo
5º) Prediccion y validacion
6º) Intrepretacion
"""

#Separamos los datos de manera aleatoria en 70% y 30%
#70 para entrenar el modelo y 30 para evaluarlo, si lo hicieramos con el 100% 
#Seria hacer trampas, ya que sabriamos el resultado.


#lo primero sera siempre hacer una copia del dataset

df_ml = df.copy()
print("Comprobamos antes que la copia del df se hizo de manera optima")
print(df_ml.info())

#Preparamos los datos para poder modelizarlos.
#NO PUEDE HABER NULOS y TODOS en NUMERICOS

#Eliminamos la columna nombre
df_ml.drop("nombre", axis=1, inplace=True)

print()

print(df_ml.info())



# Crear objetos OneHotEncoder para las columnas "elo" y "posicion"
elo_encoder = OneHotEncoder(sparse_output=False)
posicion_encoder = OneHotEncoder(sparse_output=False)

# Ajustar y transformar los datos
elo_encoded = elo_encoder.fit_transform(df_ml[['elo']])
posicion_encoded = posicion_encoder.fit_transform(df_ml[['posicion']])

# Agregar las nuevas columnas codificadas al DataFrame
elo_encoded_df = pd.DataFrame(elo_encoded, columns=elo_encoder.get_feature_names_out(['elo']))
posicion_encoded_df = pd.DataFrame(posicion_encoded, columns=posicion_encoder.get_feature_names_out(['posicion']))

# Concatenar las nuevas columnas al DataFrame original y eliminar las columnas originales
df_ml = pd.concat([df_ml, elo_encoded_df, posicion_encoded_df], axis=1)
df_ml.drop(['elo', 'posicion'], axis=1, inplace=True)


print()

print(df_ml.info())
print()

# Mostrar un muestreo aleatorio de las filas
sample = df_ml.sample(10)  # Mostrar 10 filas como ejemplo
print(sample)

# Muestra valores únicos en una columna codificada específica (por ejemplo, 'elo_Gran Maestro')
unique_values = df_ml['elo_Gran Maestro'].unique()
print(unique_values)

#La variable que intentaremos predecir sera el winrate
#Diseño de modelizacion
print("CARACOLA")
#Separacion de la variable predictoras y target
x = df_ml.drop(columns="winrate")
y  = df_ml["winrate"]

#Serapamos el train y el test

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3)



#Aqui cometi el error de intentar realizar un arbol de decisiones con datos continuos
#Por lo que me daba un error: Unknown label type: continuous
#Copiado de chat-gpt :
"""
    The error you're encountering, "Unknown label type: continuous," is because
    you're trying to train a classifier (DecisionTreeClassifier) with a target 
    variable (winrate) that appears to be continuous. Classifiers are suitable 
    for classification problems where the target variable has discrete labels 
    (e.g., classes like "class 1," "class 2," "class 3").
"""
"""
from sklearn.tree import DecisionTreeClassifier

#Instanciamos

ac= DecisionTreeClassifier(max_depth=3)

#Entrenar

ac.fit(train_x,train_y)

#Prediccion

pred = ac.predict_proba(test_x)[:,1]
pred[:20]

#Evaluacion

from sklearn.metrics import roc_auc_score

print(roc_auc_score(test_y,pred))

"""

#Aplicamos regresion lineal para intentar predecir el winrate

#Como siempre hacemos la importacion necesaria.
from sklearn.linear_model import LinearRegression


model = LinearRegression()
#Entrenamos el modelo
model.fit(train_x, train_y)

#Realizamos las predicciones
predictions = model.predict(test_x)

#Evaluacion con el error cuadratico medio
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test_y, predictions)

print("Error Cuadrático Medio (MSE):", mse)


print("Los datos al ser generados aleatoriamente, no constituyen el mejor conjunto de datos, es solo para practicar y ver que funciona ")

#Creacion de la grafica para el modelo de regrresion
#Comparamos los datos reales con los de test

plt.scatter(test_y, predictions, c='blue', label='Predicciones')
plt.scatter(test_y, test_y, c='red', label='Valores Reales')  # Puntos en la línea diagonal
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Predicciones vs. Valores Reales")
plt.legend()
plt.show()





















