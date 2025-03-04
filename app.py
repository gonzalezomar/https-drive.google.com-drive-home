import streamlit as st
import numpy as np
import pickle

# Cargar el modelo entrenado (para producción, debería guardarse y cargarse con pickle)
model = pickle.loads(b'')  # Aquí se cargaría el modelo guardado

# Opciones para cada variable
opciones_sexo = {"Hombre": 0, "Mujer": 1}
opciones_calidad = {"Rojo": 0, "Amarillo": 1, "Verde": 2}
opciones_acompanamiento = {"Solo": 0, "Acompañado": 1}
opciones_edad = {"<30": 0, "30-44": 1, "45-55": 2, "56-65": 3}
opciones_residencia = {"Asturias": 0, "Fuera de Asturias": 1}
opciones_estado_civil = {"Soltero": 0, "Casado": 1}
opciones_cargas = {"Hijos": 0, "No Hijos": 1}
opciones_situacion = {"Trabajando": 0, "Desempleo": 1, "IT": 2, "IPT": 3}
opciones_grupo = {"Grupo 1": 0, "Grupo 2": 1, "Grupo 3": 2}

# Interfaz de usuario con Streamlit
st.title("Predicción de Conversión de Clientes")

# Formulario para ingresar los datos del cliente
sexo = st.selectbox("Sexo", list(opciones_sexo.keys()))
calidad = st.selectbox("Calidad del Cliente", list(opciones_calidad.keys()))
acompanamiento = st.selectbox("Acompañamiento", list(opciones_acompanamiento.keys()))
edad = st.selectbox("Edad", list(opciones_edad.keys()))
residencia = st.selectbox("Residencia", list(opciones_residencia.keys()))
estado_civil = st.selectbox("Estado Civil", list(opciones_estado_civil.keys()))
cargas = st.selectbox("Cargas Familiares", list(opciones_cargas.keys()))
situacion = st.selectbox("Situación Laboral", list(opciones_situacion.keys()))
grupo = st.selectbox("Grupo Profesional", list(opciones_grupo.keys()))

# Botón para predecir
if st.button("Predecir Probabilidad de Contratación"):
    # Convertir inputs a valores numéricos
    datos_cliente = np.array([
        opciones_sexo[sexo],
        opciones_calidad[calidad],
        opciones_acompanamiento[acompanamiento],
        opciones_edad[edad],
        opciones_residencia[residencia],
        opciones_estado_civil[estado_civil],
        opciones_cargas[cargas],
        opciones_situacion[situacion],
        opciones_grupo[grupo]
    ]).reshape(1, -1)
    
    # Predecir la probabilidad con el modelo
    probabilidad = model.predict_proba(datos_cliente)[0][1] * 100  # Probabilidad de contratar
    
    st.write(f"Probabilidad de que el cliente contrate: **{probabilidad:.2f}%**")
