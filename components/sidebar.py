import streamlit as st

def load_sidebar():
    st.sidebar.title("Model Selection Web App")
    st.sidebar.markdown("Is your wine good quality?")

    st.sidebar.subheader("Choose Dataset")
    dataset = st.sidebar.selectbox("Dataset", ("Wine Quality Classifier"))

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    sidebar_data = {
        "dataset": dataset,
        "classifier": classifier
    }

    return sidebar_data

