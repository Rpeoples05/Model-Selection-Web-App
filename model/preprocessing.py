import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(dataset):
    if dataset == "Wine Quality Classifier":
        white_wine = pd.read_csv("data/winequality-white.csv", sep = ";")
        white_wine["type"] = "White"
        red_wine = pd.read_csv("data/winequality-red.csv", sep = ";")
        red_wine["type"] = "Red"
        df = pd.concat([white_wine, red_wine])
    if dataset == "Mushroom Edibility Classifier":
        df = pd.read_csv("data/mushrooms.csv")
    return df

def preprocess(dataset, df):
    print(df.columns)
    if dataset == "Wine Quality Classifier":
        df["category"] = "Okay"

        df.loc[df['quality'] < 4, "category"] = "Bad"
        df.loc[df['quality'] > 7, "category"] = "Good"

        df['category'] = df['category'].astype('category')
        df['type'] = df['type'].astype('category')
        df = df.drop(columns = ['quality'])

    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['float64']).columns.tolist()

    scaler = StandardScaler()
    labelencoder = LabelEncoder()

    if len(numerical_cols) > 0:
        scaled_data = scaler.fit_transform(df[numerical_cols])
        df_scaled = pd.DataFrame(scaled_data, columns = numerical_cols, index = df.index)

    for col in categorical_cols:
        df[col] = labelencoder.fit_transform(df[col])

    if len(numerical_cols) > 0:
        df = pd.concat([df_scaled, df[categorical_cols]], axis=1)

    return df

def split(df, dataset):
    if dataset == "Wine Quality Classifier":
        y = df.category
        X = df.drop(columns = ['category'])
    if dataset == "Mushroom Edibility Classifier":
        y = df.type
        X = df.drop(columns = ['type'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42, stratify = y)
    return X_train, X_test, y_train, y_test
