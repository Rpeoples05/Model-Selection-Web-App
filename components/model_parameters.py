import streamlit as st
from components import config

def svm_parameters(dataset):
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
    kernel = st.sidebar.radio("Kernel", ("rbf","linear"), key='kernel')
    gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

    if len(config.CLASS_NAMES[dataset]) > 2:
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "Classification Report"))
    else:
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Classification Report"))

    return C, kernel, gamma, metrics

def logistic_parameters(dataset):
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key = 'C_LR')
    max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

    if len(config.CLASS_NAMES[dataset]) > 2:
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "Classification Report"))
    else:
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Classification Report"))

    return C, max_iter, metrics

def forest_parameters(dataset):
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
    max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key='bootstrap')

    if len(config.CLASS_NAMES[dataset]) > 2:
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "Classification Report"))
    else:
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Classification Report"))

    return n_estimators, max_depth, bootstrap, metrics