import streamlit as st
import pandas as pd
import joblib

#Configuración inicial de la página de la aplicación web
st.set_page_config(page_title="Predicción depósito bancario")

#El decorador cache_resource permite cargar el modelo en memoria sólo una vez
@st.cache_resource
def cargar_modelo_y_encoder():
    """Carga el modelo de machine learning y el codificador de etiquetas desde los archivos."""
    modelo = joblib.load("modelo_final.joblib")
    encoder = joblib.load("label_encoder.pkl")
    return modelo, encoder

def transformar_pdays(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la misma transformación a la variable 'pdays' que usamos en el proceso de entrenamiento."""
    df = df.copy()
    df["contacted_before"] = (df["pdays"] > -1).astype(int)
    df["pdays"] = df["pdays"].apply(lambda x: 0 if x == -1 else x)
    return df

#Carga de los recursos en memoria al arrancar la app
modelo, le = cargar_modelo_y_encoder()

st.title("Predicción de suscripción a depósito")
st.write("Introduce los datos de un cliente para predecir si contratará el depósito.")

#Creación de un formulario de entrada de datos para el cliente
with st.form("form_cliente"):
    age = st.number_input("Edad", min_value=18, max_value=100, value=40, step=1)

    job = st.selectbox("Trabajo", [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ])

    marital = st.selectbox("Estado civil", ["divorced", "married", "single"])
    education = st.selectbox("Educación", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("¿Tiene créditos impagados?", ["no", "yes"])
    balance = st.number_input("Balance anual medio", value=1000, step=100)
    housing = st.selectbox("¿Tiene hipoteca?", ["no", "yes"])
    loan = st.selectbox("¿Tiene préstamo personal?", ["no", "yes"])
    contact = st.selectbox("Tipo de contacto", ["cellular", "telephone", "unknown"])
    day = st.number_input("Día del último contacto", min_value=1, max_value=31, value=15, step=1)
    month = st.selectbox("Mes del último contacto", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])
    duration = st.number_input("Duración del último contacto (segundos)", min_value=0, value=180, step=10)
    campaign = st.number_input("Número de contactos en esta campaña", min_value=1, value=1, step=1)

    pdays_opcion = st.selectbox(
        "¿Fue contactado en la campaña anterior?",
        ["No / desconocido", "Sí"]
    )

    if pdays_opcion == "No / desconocido":
        pdays = -1
    else:
        pdays = st.number_input(
            "Días desde el contacto anterior",
            min_value=0,
            value=10,
            step=1
        )

    previous = st.number_input("Número de contactos previos", min_value=0, value=0, step=1)
    poutcome = st.selectbox("Resultado de campañas anteriores", ["failure", "other", "success", "unknown"])

    #Se genera el botón que lanzará la predicción
    submitted = st.form_submit_button("Predecir")

#Acción a ejecutar en el momento en el que se envía el formulario
if submitted:
    #Se crea un DataFrame de Pandas con los valores rellenados por el usuario
    nueva_instancia = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    #Se aplica la transformación a los datos según espera el modelo de Machine Learning
    nueva_instancia = transformar_pdays(nueva_instancia)

    #Realizar la predicción numérica y luego usar el inverse_transform para sacar el valor textual (e.g. 'yes' / 'no')
    pred = modelo.predict(nueva_instancia)
    pred_label = le.inverse_transform(pred)[0]

    #Mostrar la interfaz del resultado en base al label descifrado
    st.subheader("Resultado")
    if pred_label == "yes":
        st.success("El modelo predice que el cliente SÍ suscribirá el depósito.")
    else:
        st.error("El modelo predice que el cliente NO suscribirá el depósito.")

    st.write("Datos introducidos:")
    st.dataframe(nueva_instancia)