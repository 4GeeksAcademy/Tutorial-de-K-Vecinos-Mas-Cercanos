from utils import db_connect
engine = db_connect()

import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# URLs de los conjuntos de datos
movies_url = 'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv'
credits_url = 'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv'

# Cargar los datos en DataFrames
movies_df = pd.read_csv(movies_url)
credits_df = pd.read_csv(credits_url)

# Mostrar las primeras filas de los DataFrames
print("Películas:")
print(movies_df.head())
print("\nCréditos:")
print(credits_df.head())

# Unir los DataFrames usando las columnas adecuadas
merged_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')

# Seleccionar las columnas necesarias
merged_df = merged_df[['id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']]
merged_df.rename(columns={'title_x': 'title'}, inplace=True)

# Mostrar las primeras filas del DataFrame combinado
print("\nDatos Combinados:")
print(merged_df.head())

# Función para convertir las columnas JSON
def convert_json_column(column):
    return column.apply(lambda x: [i['name'] for i in ast.literal_eval(x)])

# Convertir las columnas genres y keywords
merged_df['genres'] = convert_json_column(merged_df['genres'])
merged_df['keywords'] = convert_json_column(merged_df['keywords'])

# Convertir la columna cast y seleccionar los tres primeros nombres
merged_df['cast'] = merged_df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])

# Convertir la columna crew y seleccionar el nombre del director
def get_director(crew):
    for i in ast.literal_eval(crew):
        if i['job'] == 'Director':
            return i['name']
    return ''

merged_df['crew'] = merged_df['crew'].apply(get_director)

# Limpiar espacios en blanco en las columnas
columns_to_clean = ['genres', 'keywords', 'cast', 'crew']
for column in columns_to_clean:
    merged_df[column] = merged_df[column].apply(lambda x: ' '.join(x) if isinstance(x, list) else x.replace(' ', ''))

# Crear la columna 'tags'
merged_df['tags'] = merged_df.apply(lambda row: f"{row['overview']} {' '.join(row['genres'])} {' '.join(row['keywords'])} {' '.join(row['cast'])} {row['crew']}", axis=1)

# Seleccionar solo las columnas necesarias
final_df = merged_df[['id', 'title', 'tags']]

# Mostrar la tabla final
print("\nDatos Finales:")
print(final_df.head())

# Vectorizar la columna 'tags'
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(final_df['tags']).toarray()

# Calcular la similitud del coseno
similarity = cosine_similarity(vectors)

# Función de recomendación basada en similitud del coseno
def recommend(movie):
    try:
        movie_index = final_df[final_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        print("Películas recomendadas:")
        for i in movie_list:
            print(final_df.iloc[i[0]].title)
    except IndexError:
        print("La película no se encuentra en la base de datos.")

# Probar la función de recomendación
recommend("Spectre")

# Visualización 3D de géneros y palabras clave
total_data = merged_df.copy()
total_data['genre_count'] = total_data['genres'].apply(lambda x: len(x))
total_data['keyword_count'] = total_data['keywords'].apply(lambda x: len(x))

fig = px.scatter_3d(total_data, x='genre_count', y='keyword_count', z='id', color='title', width=1000, height=500,
                    size=total_data['genre_count'].abs())
camera = dict(
    up=dict(x=1, y=3.5, z=0),
    eye=dict(x=2, y=0, z=0)
)

fig.update_layout(scene_camera=camera)
fig.show()



