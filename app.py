import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
from model import preprocessing
from components import sidebar
from model import models

def main():
    st.title("Model Selection Web App")
    st.markdown("Is your wine good quality?")
    sidebar_data = sidebar.load_sidebar()

    @st.cache_data(persist = True)
    def load_process(dataset):
        data = preprocessing.load_data(dataset)
        preprocessing.preprocess(dataset, data)

    @st.cache_data(persist = True)
    def split(df):
        X_train, X_test, y_train, y_test = preprocessing.split(df)
        return X_train, X_test, y_train, y_test
    
    df = load_process(sidebar_data["dataset"])
    X_train, X_test, y_train, y_test = split(df)

    if sidebar_data["classify"]:
        if sidebar_data['classifier'] == "Support Vector Machine (SVM)":
            model = models.svm_model(X_train, y_train, sidebar_data["hyperparameters"])
            accuracy, precision, recall = models.model_predictions(model, X_test, y_test)

        if sidebar_data['classifier'] == "Logistic Regression":
            model = models.logistic_model(X_train, y_train, sidebar_data['hyperparameters'])
            accuracy, precision, recall = models.model_predictions(model, X_test, y_test)

        if sidebar_data['classifier'] == "Random Forest":
            model = models.forest_model(X_train, y_train, sidebar_data['hyperparameters'])
            accuracy, precision, recall = models.model_predictions(model, X_test, y_test)
            
if __name__ == '__main__':
    main()