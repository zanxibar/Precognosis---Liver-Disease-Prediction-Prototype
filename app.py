# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:23:53 2024

@author: user
"""

import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Load the trained model and training data
model = load_model('best_liver_disease_model')
train_df = joblib.load('liver_train_data.pkl')

# Function to get user input
def get_user_input():
    age = st.number_input('Age', min_value=0, max_value=120, value=0)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    total_bilirubin = st.number_input('Total Bilirubin', min_value=0.0, max_value=100.0, value=0.0)
    direct_bilirubin = st.number_input('Direct Bilirubin', min_value=0.0, max_value=100.0, value=0.0)
    alkaline_phosphotase = st.number_input('Alkaline Phosphotase', min_value=0, max_value=2000, value=0)
    alamine_aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0, max_value=2000, value=0)
    aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0, max_value=2000, value=0)
    total_proteins = st.number_input('Total Proteins', min_value=0.0, max_value=10.0, value=0.0)
    albumin = st.number_input('Albumin', min_value=0.0, max_value=10.0, value=0.0)
    albumin_and_globulin_ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0, max_value=5.0, value=0.0)
    
    # Convert gender to numerical value
    gender = 1 if gender == 'Male' else 0
    
    user_data = {
        'Age': age,
        'Gender': gender,
        'Total_Bilirubin': total_bilirubin,
        'Direct_Bilirubin': direct_bilirubin,
        'Alkaline_Phosphotase': alkaline_phosphotase,
        'Alamine_Aminotransferase': alamine_aminotransferase,
        'Aspartate_Aminotransferase': aspartate_aminotransferase,
        'Total_Protiens': total_proteins,
        'Albumin': albumin,
        'Albumin_and_Globulin_Ratio': albumin_and_globulin_ratio
    }
    
    return pd.DataFrame(user_data, index=[0])

# Prediction Page
def prediction():
    st.title("Liver Disease Prediction")
    user_input = get_user_input()
    
    if st.button("Predict"):
        prediction_result = predict_model(model, data=user_input)
        
        if 'prediction_label' in prediction_result.columns and 'prediction_score' in prediction_result.columns:
            predicted_class = int(prediction_result['prediction_label'].iloc[0])
            predicted_prob = prediction_result['prediction_score'].iloc[0]
            
            st.write("Prediction Result:")
            st.write(f"Predicted Class: {'Liver Disease' if predicted_class == 1 else 'No Liver Disease'}")
            st.write(f"Probability: {predicted_prob:.4f}")
            
            # Explanation of the prediction
            st.subheader("Prediction Explanation")
            if predicted_class == 1:
                st.write("The model predicts that the patient is likely to have liver disease.")
            else:
                st.write("The model predicts that the patient is unlikely to have liver disease.")
            
            st.subheader("Probability Explanation")
            st.write("""
                The probability score indicates the model's confidence in its prediction. 
                A higher score closer to 1 means the model is more confident that the patient has liver disease, 
                while a score closer to 0 means the model is more confident that the patient does not have liver disease.
            """)
            
            # Risk Factor Analysis using SHAP
            st.subheader("Risk Factor Analysis")
            shap.initjs()

            # Explaining the model's predictions using SHAP
            explainer = shap.Explainer(model.predict, train_df.drop(columns=['Dataset']))
            shap_values = explainer(user_input)
            
            st.write("SHAP Values for Risk Factors:")
            st.write(shap_values.values)

            # Feature Importance Plot
            st.subheader("Feature Importance")
            st.write("""
                The Feature Importance plot shows the most significant factors contributing to the prediction.
                The higher the bar, the more influence that feature has on the model's prediction.
            """)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, user_input, plot_type="bar", show=False)
            st.pyplot(fig)

            # SHAP Summary Plot
            st.subheader("SHAP Summary Plot")
            st.write("""
                The SHAP Summary plot provides a visualization of the impact of each feature on the model's output.
                Each dot represents a feature's impact on the prediction for a specific instance. 
                The color represents the value of the feature (red for high, blue for low).
            """)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, user_input, show=False)
            st.pyplot(fig)

            # SHAP Force Plot
            st.subheader("SHAP Force Plot")
            st.write("""
                The SHAP Force plot illustrates the impact of each feature on the model's prediction for a single instance.
                Features pushing the prediction towards a higher value (indicating liver disease) are shown in red, 
                while those pushing towards a lower value (indicating no liver disease) are in blue.
            """)
            st_shap(shap.force_plot(shap_values[0]), height=400)

            # SHAP Values Explanation
            st.subheader("SHAP Values Explanation")
            st.write("""
                SHAP values help in understanding the model's prediction by showing the contribution of each feature.
                - Positive SHAP values indicate that the feature contributes to predicting a higher probability of liver disease.
                - Negative SHAP values indicate that the feature contributes to predicting a lower probability of liver disease.
                - The magnitude of the SHAP value shows the strength of the contribution.
            """)
            
        else:
            st.error("Error: Prediction did not return 'prediction_label' or 'prediction_score' columns.")

# Helper function to display SHAP force plot in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Main function
def main():
    prediction()

if __name__ == "__main__":
    main()
