import streamlit as st
from components import model_parameters

def load_sidebar():
    st.sidebar.title("Model Selection Web App")
    st.sidebar.markdown("Is your wine good quality?")

    st.sidebar.subheader("Choose Dataset")
    dataset = st.sidebar.selectbox("Dataset", ("Wine Quality Classifier", "Mushroom Edibility Classifier"))

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        C, kernel, gamma, metrics = model_parameters.svm_parameters(dataset)
        hyperparameters = {
            "C": C,
            "kernel": kernel,
            "gamma": gamma,
        }
    
    if classifier == "Logistic Regression":
        C, max_iter, metrics = model_parameters.logistic_parameters(dataset)
        hyperparameters = {
            "C": C,
            "max_iter": max_iter
        }
    
    if classifier == "Random Forest":
        n_estimators, max_depth, bootstrap, metrics = model_parameters.forest_parameters(dataset)
        hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "bootstrap": bootstrap
        }
    
    classify = st.sidebar.button("Classify", key='classify')

    sidebar_data = {
        "dataset": dataset,
        "classifier": classifier,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "classify": classify
    }

    return sidebar_data

