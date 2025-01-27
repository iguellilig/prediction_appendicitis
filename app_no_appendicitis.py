import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('Classifying paediatric appendicitis')
st.markdown('Classification model to classify paediatric into \
     (diagnosed/ not diagnosed) based on different features.')

st.header("Patient Features")
col1, col2 = st.columns(2)

with col1:
    age = st.text_input('Age: ', '10.15742642')
    bmi = st.text_input('BMI: ', '21.01792013')
    gender = st.radio("Sex",options=["Male", "Female", "No_value"], index=1)
    height = st.text_input('Height: ', '142.2')
    weight = st.text_input('Weight: ', '42.5')
    appendix_diameter = st.text_input('Appendix_Diameter: ', '')
    migratory_pain = st.radio("Migratory_Pain",options=["Yes", "No", "No_value"], index=0)
    lower_right_abd_pain = st.radio("Lower_Right_Abd_Pain",options=["Yes", "No", "No_value"], index=0)
    contralateral_rebound_tenderness = st.radio("Contralateral_Rebound_Tenderness",options=["Yes", "No", "No_value"], index=0)
    coughing_pain = st.radio("Coughing_Pain",options=["Yes", "No"], index=1)
    loss_of_appetite = migratory_pain = st.radio("Loss_of_Appetite",options=["Yes", "No", "No_value"],index=1)
    body_temprerature = st.text_input('Body_Temperature: ', '37.4')
    wbc_count = st.text_input('WBC_Count: ', '15')
    neutrophil_percentage = st.text_input('Neutrophile_Percentage: ', '72.5')    
    segmented_neutrophils = st.text_input('Segmented_Neutrophils: ', '')
    neutrophilia = st.radio("Neutrophilia",options=["Yes", "No", "No_value"], index=1)
with col2:
    rbc_count = st.text_input('RBC_Count: ', '4.73')
    hemoglibin = st.text_input('Hemoglobin: ', '12.5')
    rdw = st.text_input('RDW: ', '12.9')
    thrombocyte_count = st.text_input('Thrombocyte_Count: ', '318')
    ketones_in_urine = st.radio("Ketones_in_Urine",options=["No", "+", "++", "+++", "No_value"], index=0)
    rbc_in_urine = st.radio("RBC_in_Urine",options=["No", "+", "++", "+++", "No_value"], index=0)
    wbc_in_urine = st.radio("WBC_in_Urine",options=["No", "+", "++", "+++", "No_value"], index=0)
    crp = st.text_input('CRP: ', '2')
    dysuria = st.radio("Dysuria",options=["Yes", "No"], index=1)
    stool = st.radio("Stool",options=["Normal", "Constipation", "Diarrhea", "Constipation,Diarrhea", "No_value"], index=0)
    peritonitis = st.radio("Peritonitis",options=["No", "Local", "Generalized", "No_value"], index=0)
    psoas_sign = st.radio("Psoas_Sign",options=["Yes", "No", "No_value"], index=1)


list_non_string= [age,
              bmi,
              height,
              weight,
              appendix_diameter,
              body_temprerature,
              wbc_count,
              neutrophil_percentage,
              segmented_neutrophils,
              rbc_count,
              hemoglibin,
              rdw,
              thrombocyte_count,
              crp]

for elem in range(len(list_non_string)):
    if list_non_string[elem] == "":
        list_non_string[elem] = "-1"

list_values= [list_non_string[1],
              gender,
              list_non_string[2],
              list_non_string[3],
              list_non_string[4],
              migratory_pain, 
              lower_right_abd_pain,
              contralateral_rebound_tenderness,
              coughing_pain,
              loss_of_appetite,
              list_non_string[5],
              list_non_string[6],
              list_non_string[7],
              list_non_string[8],
              neutrophilia,
              list_non_string[9],
              list_non_string[10],
              list_non_string[11],
              list_non_string[12],
              ketones_in_urine,
              rbc_in_urine,
              wbc_in_urine,
              list_non_string[13],
              dysuria,
              stool,
              peritonitis,
              psoas_sign]

results = list_non_string[0]
for elem in list_values:
    results = results + " " + elem

list_string_elem = [gender,
              migratory_pain, 
              lower_right_abd_pain,
              contralateral_rebound_tenderness,
              coughing_pain,
              loss_of_appetite,
              neutrophilia,
              ketones_in_urine,
              rbc_in_urine,
              wbc_in_urine,
              dysuria,
              stool,
              peritonitis,
              psoas_sign]

map = {"No_value":-1, 'Male':0, 'Female':1, 'Normal': 7, 'Yes': 2, 'No':3, '+':4, '++':5, '+++':6, 'Constipation,Diarrhea':10, 'Constipation':8, 'Diarrhea':9, 'Local':11, 'Generalized':2}

for elem in map:
    if elem in results:
        results = results.replace(elem, str(map[elem]))


list_results = results.split(" ")
if st.button("Diagnosis of paediatric appendicitis"):
    result = predict(
        np.array([list_results]))
    if result[0] == 0:
        st.text("No appendicitis")
    elif result[0] == 1:
        st.text("Appendicitis")
    else:
       st.text(result)
       st.text(result[0])




        
