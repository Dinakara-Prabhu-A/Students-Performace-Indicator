import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

def predict_datapoint():
    st.title("Student Performance Predictor")
    st.subheader("Enter the details of the student to predict their performance.")

    gender = st.selectbox("Select the gender :",['female' ,'male'])
    
    race_ethnicity = st.selectbox("Select the race/ethnicity :", ['group A' ,'group B', 'group C' ,'group D' ,'group E'])

    parental_level_of_education = st.selectbox("Select the parental level of education :", ["bachelor's degree", 'some college', "master's degree",
       "associate's degree", 'high school', 'some high school'])
    
    lunch = st.selectbox("Select the lunch type :", ['standard', 'free/reduced'])
    test_preparation_course = st.selectbox("Select the test preparation course :", ['none' ,'completed'])
    
    reading_score = st.number_input("Enter the reading score :",1,100,1)
    writing_score = st.number_input("Enter the writing score :",1,100,1)
    
    if gender and race_ethnicity and parental_level_of_education and lunch and test_preparation_course and reading_score and writing_score:

        if st.button("predict"):

            data = CustomData(gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score)
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            st.subheader("Predicted Performance based on the entered details:")
            st.write("prediction of math score ",results)
            return results
    


predict = predict_datapoint()