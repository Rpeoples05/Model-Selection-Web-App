import streamlit as st
import pandas as pd
from components import config
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, classification_report

def plot_metrics(metrics_list, model, x_test, y_test, y_pred, dataset):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=config.CLASS_NAMES[dataset], ax=ax)
        st.pyplot(fig)

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)
    
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)
    
    if "Classification Report" in metrics_list:
        st.subheader('Classification Report')
        report = classification_report(y_test, y_pred, target_names = config.CLASS_NAMES[dataset], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

def write_scores(accuracy, precision, recall):
    st.write("Accuracy: ", accuracy)
    st.write("Precision: ", precision)
    st.write("Recall: ", recall)