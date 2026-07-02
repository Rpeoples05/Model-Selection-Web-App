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

def main():
    st.title("Model Selection Web App")
    st.markdown("Is your wine good quality?")

    @st.cache_data(persist = True)
    def load_process(dataset):
        data = preprocessing.load_data(dataset)
        preprocessing.preprocess(dataset, data)

    df = load_process("Wine Quality Classifier")

if __name__ == '__main__':
    main()