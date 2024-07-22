# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:13:27 2024

@author: User
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#loading saved model
diabetesModel=pickle.load(open('C:/Users/User/OneDrive/Desktop/multipleDiseasePrediction/savedModels/trainedModel_RF.sav','rb'))
heartDiseaseModel=pickle.load(open('C:/Users/User/OneDrive/Desktop/multipleDiseasePrediction/savedModels/heartTrainedModel.sav','rb'))
parkinsonModel=pickle.load(open('C:/Users/User/OneDrive/Desktop/multipleDiseasePrediction/savedModels/parkinsonTrainedModel.sav','rb'))

#sidebar for navigation
with st.sidebar:
    selected=option_menu('Multiple Disease Prediction System',
                         ['Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction'],icons=['activity','heart','person'],default_index=0
                         )

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction Model')
    # Add your diabetes prediction model code here
    col1,col2,col3=st.columns(3)
    with col1:
        pregnancies=st.text_input('Number of pregnancies')
    with col2:
        glucose=st.text_input('Glucose Level')
    with col3:
        bp=st.text_input('Blood Pressure (BP)')
    with col1:
        skinThickness=st.text_input('Skin Thickness')
    with col2:
        insulin=st.text_input('Insulin Level')
    with col3:
        bmi=st.text_input('Body-Mass Index (BMI)')
    with col1:
        diabetesFunc=st.text_input('Diabetes Pedigree Function value')
    with col2:
        age=st.text_input('Age')
    
    #code for prediction
    dataContainer=[pregnancies,glucose,bp,skinThickness,insulin,bmi,diabetesFunc,age]
    diabetesDiagnosis=""
    #button to predict
    if st.button('Predict the Diabetes Result'):
        diabetesPrediction=diabetesModel.predict([dataContainer])
        if(diabetesPrediction[0]==1):
            diabetesDiagnosis='The person is diabetic'
        else:
            diabetesDiagnosis='The person is not diabetic'
    st.success(diabetesDiagnosis)

elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction Model')
    # Add your heart disease prediction model code here
    col1,col2,col3=st.columns(3)
    with col1:
        age=st.text_input('Age')
    with col2:
        sex=st.text_input('Gender')
    with col3:
        cp=st.text_input('Cerebral Palsy')
    with col1:
        trestbps=st.text_input('Resting Blood Pressure')
    with col2:
        chol=st.text_input('Cholestrol Level')
    with col3:
        fbs=st.text_input('Fasting Blood Sugar')
    with col1:
        ecg=st.text_input('Electrocardiogram')
    with col2:
        thalach=st.text_input('The person maximum heart rate achieved')
    with col3:
        exang=st.text_input('Exang')
    with col1:
        oldpeak=st.text_input('ST depression induced by excercise relative to rest (oldpeak)')
    with col2:
        slope=st.text_input('Slope of Recovery')
    with col3:
        ca=st.text_input('Cardiac Arrest')
    with col1:
        thal=st.text_input('Thalassemia')
    with col2:
        target=st.text_input('Target')
    #code for prediction
    dataContainer2=[age,sex,cp,trestbps,chol,fbs,ecg,thalach,exang,oldpeak,slope,ca,thal,target]
    heartDiagnosis=""
    #button to predict
    if st.button('Predict the Heart Disease Result'):
        heartPrediction=heartDiseaseModel.predict([dataContainer2])
        if(heartPrediction[0]==1):
            heartDiagnosis='The person has a heart disease'
        else:
            heartDiagnosis='The person does not have a heart disease'
    st.success(heartDiagnosis)

elif selected == 'Parkinsons Prediction':
    st.title('Parkinsons Prediction Model')
    # Add your Parkinson's prediction model code here
    col1,col2,col3=st.columns(3)
    with col1:
        fo=st.text_input('Foramen Ovale')
    with col2:
        flo=st.text_input('Fluorouracil-leucovorin-Oxaliplatin (FLO)')
    with col3:
        jitter=st.text_input('MDVP-Jitter(%)')
    with col1:
        jitter1=st.text_input('MDVP-Jitter(abs)')
    with col2:
        rap=st.text_input('MDVP-RAP')
    with col3:
        ppq=st.text_input('MDVP-PPQ')
    with col1:
        ddpJitter=st.text_input('DDP')
    with col2:
        shimmer=st.text_input('MDVP-Shimmer')
    with col3:
        shimmer1=st.text_input('MDVP-Shimmer(db)')
    with col1:
        shimmer2=st.text_input('APQ3')
    with col2:
        shimmer3=st.text_input('APQ5')
    with col3:
        shimmer4=st.text_input('MDVP-APQ')
    with col1:
        shimmer5=st.text_input('MDVP-Shimmer2')
    with col2:
        dda=st.text_input('DDA')
    with col3:
        nhr=st.text_input('NHR')
    with col1:
        hnr=st.text_input('HNR')
    with col2:
        status=st.text_input('Status')
    with col3:
        rpde=st.text_input('RPDE')
    with col1:
        dfa=st.text_input('DFA')
    with col2:
        spread1=st.text_input('Spread1')
    with col3:
        spread2=st.text_input('Spread2')
    with col1:
        d2=st.text_input('D2')
    with col2:
        age=st.text_input('Age')
    with col3:
        ppe=st.text_input('PPE')

#code for prediction
dataContainer3=[fo,flo,jitter,jitter1,rap,ppq,ddpJitter,shimmer,shimmer1,shimmer2,shimmer3,shimmer4,shimmer4,shimmer5,dda,nhr,hnr,status,rpde,dfa,spread1,spread2,d2,ppe]
parkinsonDiagnosis=""
#button to predict
if st.button('Predict the Parkinson Disease Result'):
    parkinsonPrediction=parkinsonModel.predict([dataContainer3])
    if(parkinsonPrediction[0]==1):
        parkinsonDiagnosis='The person has a heart disease'
    else:
        parkinsonDiagnosis='The person does not have a heart disease'
st.success(parkinsonDiagnosis)
    