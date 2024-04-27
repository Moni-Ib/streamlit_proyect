import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.title('**Proyecto Programación**')
st.markdown('**Mónica Ibarra Herrera**')
st.image('https://e00-elmundo.uecdn.es/assets/multimedia/imagenes/2022/12/07/16704111377160.jpg')
url_youtube = "https://www.youtube.com/watch?v=RircTZnd3Zg"
st.video(url_youtube)

st.header("Data set")
df=pd.read_csv("C:/Users/cesar/apps/proyecto_data_analytics/proyecto_data_analytics/notebooks/df_corregido.csv") 
st.dataframe(df)
print('\n')

st.header("Partidos de México")
df_mexico = pd.read_csv("C:/Users/cesar/apps/proyecto_data_analytics/proyecto_data_analytics/notebooks/partidos_mex.csv") 
df_mexico.columns=["year","country","city","stage","home_team","away_team","home_score","away_score","outcome","winning_team","losing_team","date", "month", "dayofweek", 'score_mexico', 'opponent_mex'] 
st.dataframe(df_mexico)
print('\n')

#Visualizaciones países
st.header('Finales mundiales')
finales = df[df['stage']=='Final']
st.dataframe(finales)

st.header("Mundiales ganados por países")
finales= df[df['stage']=='Final']
ganadores_finales = finales['winning_team'].value_counts()
ganadores_finales
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=ganadores_finales.index, y=ganadores_finales.values, palette="viridis", ax=ax)
plt.title('Mundiales ganados por países')
plt.xlabel('Países')
plt.ylabel('# de Mundiales')
plt.xticks(rotation=45)
st.pyplot(fig)
st.image('https://www.mundodeportivo.com/us/files/og_thumbnail/uploads/2022/05/05/62743e4bc0a52.jpeg')
st.markdown("**Brasil es el país que ha ganado más mundiales, superando a los demás países con 5 victorias en total**")
st.markdown('Brasil ha jugado 109 partidos en los mundiales, de los cuales ha ganado 76')
#Visualizaciones México

st.header("Visualizaciones de México")
st.image('https://cdn2.mediotiempo.com/uploads/media/2018/07/02/memes-brasil-vs-mexico-adios-14.jpg')
st.markdown('México ha jugado 57 partidos en los mundiales, de los cuales sólo ha ganado 16')
ganados=df[df['winning_team']=='Mexico']
st.dataframe(ganados)
st.markdown('México ha ganado el 28.07 % de sus partidos')
mexico_home_goles = df_mexico[df_mexico['home_team'] == 'Mexico'].groupby('year')['home_score'].sum()
mexico_away_goles = df_mexico[df_mexico['away_team'] == 'Mexico'].groupby('year')['away_score'].sum()
mexico_total_goles = mexico_home_goles.add(mexico_away_goles, fill_value=0)
goles = df_mexico.groupby('year')[['home_score', 'away_score']].sum().sum(axis=1)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=mexico_total_goles.index, y=mexico_total_goles.values, palette="viridis", ax=ax)
plt.title('Goles de México en los mundiales')
plt.xlabel('Año')
plt.ylabel('# de Goles')
plt.xticks(rotation=45)
st.pyplot(fig)

mexico_goals_home = df[df['home_team'] == 'Mexico']['home_score']
mexico_goals_away = df[df['away_team'] == 'Mexico']['away_score']
total_goles_mex = mexico_goals_home.sum() + mexico_goals_away.sum()
promedio_goles_mexico = total_goles_mex / len(df_mexico)

st.markdown("Promedio de goles metidos por México: {:.2f}".format(promedio_goles_mexico))
st.markdown('México ha metido más goles en el mundial de 1998 el cual fue en Francia')
mundial_1998 = df[(df['year'] == 1998) & ((df['home_team'] == 'Mexico') | (df['away_team'] == 'Mexico'))]
st.dataframe(mundial_1998)

st.subheader('**Goles de México como local**')
mexico_local = df_mexico[df_mexico['home_team'] == 'Mexico']['home_score']
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(mexico_local)
ax.set_title('Goles de México como Local')
ax.set_ylabel('Goles')
st.pyplot(fig)
st.markdown('**México metió 4 goles cuando jugó contra el Salvador. El Salvador no ha ganado ningún partido**')
st.markdown('**México metió 3 goles cuando jugó contra Irán. Irán ha ganado sólo 2 partidos en los 21 mundiales**')

st.subheader('**Goles de México como visitante**')
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(mexico_goals_away)
ax.set_title('Goles de México como visitante')
ax.set_ylabel('Goles')
st.pyplot(fig)

st.subheader('México en las etapas de los mundiales')
participacion = df_mexico['stage'].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=participacion.index, y=participacion.values, palette="viridis", ax=ax)
ax.set_title('Frecuencia de Participación de México en Diferentes Etapas del Torneo')
ax.set_xlabel('Etapa del Torneo')
ax.set_ylabel('Partidos')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)
st.markdown('**México ha llegado hasta cuartos de final 2 veces**')

st.subheader('Cuartos de final México')
st.image('https://www.sopitas.com/wp-content/uploads/2018/06/dest-mexico2026.jpg')
partidos_mexico_cuartos = df[(df['stage']=='Quarterfinals') & ((df['home_team'] == 'Mexico') | (df['away_team']=='Mexico'))]
st.dataframe(partidos_mexico_cuartos)
st.markdown('**Los 2 cuartos de finales, han sido en México**')
st.image('https://www.planamayor.com.mx/wp-content/uploads/2016/06/meme_selmex_planamayor4.jpe')


st.header('Modelo de predicción Regresión Lineal')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
X = df_mexico[['opponent_mex']]
y = df_mexico[['score_mexico']]
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X).toarray()
scaler = MinMaxScaler()
countries_normalized = scaler.fit_transform(X_encoded)
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(countries_normalized, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
st.subheader('Oponentes de México')
df_oponentes = pd.read_csv('C:/Users/cesar/apps/proyecto_data_analytics/proyecto_data_analytics/notebooks/oponentes.csv')
st.dataframe(df_oponentes)
def predecir_goles(oponente):
    new_df = pd.DataFrame({'opponent_mex': [oponente]})
    # Codificar el nuevo dato
    new_data_encoded = encoder.transform(new_df[['opponent_mex']]).toarray()
    predicted_goals = model.predict(new_data_encoded)
    return int(predicted_goals.item())
st.subheader('Predicción de Goles de México')
oponente = st.text_input('Nombre del Oponente', 'South Africa')
if st.button('Predecir Goles'):
    goles_predichos = predecir_goles(oponente)
    st.write(f'Goles predecidor de México contra {oponente}: {goles_predichos}')
st.image('https://www.elgrafico.com/export/sites/prensagrafica/img/2018/06/06/meme6.jpg_1081867746.jpg')