from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

def svm_model(X_train, y_train, hyperparameters):
    model = SVC(C = hyperparameters["C"], kernel = hyperparameters["kernel"], gamma = hyperparameters["gamma"], class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def model_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = round(model.score(X_test, y_test),2)
    precision = round(precision_score(y_test, y_pred, average='micro'),2)
    recall = round(recall_score(y_test, y_pred, average='micro'),2)
    return accuracy, precision, recall, y_pred

def logistic_model(X_train, y_train, hyperparameters):
    model = LogisticRegression(C=hyperparameters["C"], penalty='l2', max_iter=hyperparameters["max_iter"], class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def forest_model(X_train, y_train, hyperparameters):
    model = RandomForestClassifier(n_estimators=hyperparameters["n_estimators"],max_depth = hyperparameters["max_depth"], bootstrap = hyperparameters["bootstrap"], n_jobs = -1, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

