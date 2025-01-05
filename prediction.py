import joblib


def predict(data):
    clf = joblib.load("best_model.sav")
    return clf.predict(data)
