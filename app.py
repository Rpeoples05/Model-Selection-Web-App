import streamlit as st
from model import preprocessing
from components import sidebar
from model import models
from components import evaluation

def main():
    st.title("Model Selection Web App")
    st.markdown("Is your wine good quality?")
    sidebar_data = sidebar.load_sidebar()

    @st.cache_data(persist = True)
    def load_process(dataset):
        data = preprocessing.load_data(dataset)
        data = preprocessing.preprocess(dataset, data)
        return data

    @st.cache_data(persist = True)
    def split(df):
        X_train, X_test, y_train, y_test = preprocessing.split(df)
        return X_train, X_test, y_train, y_test
    
    df = load_process(sidebar_data["dataset"])
    X_train, X_test, y_train, y_test = split(df)

    if sidebar_data["classify"]:
        if sidebar_data['classifier'] == "Support Vector Machine (SVM)":
            model = models.svm_model(X_train, y_train, sidebar_data["hyperparameters"])
            accuracy, precision, recall, y_pred = models.model_predictions(model, X_test, y_test)
            evaluation.write_scores(accuracy, precision, recall)
            evaluation.plot_metrics(sidebar_data['metrics'], model, X_test, y_test, y_pred, sidebar_data["dataset"])

        if sidebar_data['classifier'] == "Logistic Regression":
            model = models.logistic_model(X_train, y_train, sidebar_data['hyperparameters'])
            accuracy, precision, recall, y_pred = models.model_predictions(model, X_test, y_test)
            evaluation.write_scores(accuracy, precision, recall)
            evaluation.plot_metrics(sidebar_data['metrics'], model, X_test, y_test, y_pred, sidebar_data["dataset"])

        if sidebar_data['classifier'] == "Random Forest":
            model = models.forest_model(X_train, y_train, sidebar_data['hyperparameters'])
            accuracy, precision, recall, y_pred = models.model_predictions(model, X_test, y_test)
            evaluation.write_scores(accuracy, precision, recall)
            evaluation.plot_metrics(sidebar_data['metrics'], model, X_test, y_test, y_pred, sidebar_data["dataset"])

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Dataset: ", sidebar_data["dataset"])
        st.write(df)

if __name__ == '__main__':
    main()