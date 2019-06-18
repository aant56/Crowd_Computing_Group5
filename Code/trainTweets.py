import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import joblib

files = ('for.csv', 'against.csv', 'positive.csv', 'negative.csv','factual.csv', 'emotional.csv' )
ml_models = ('for.sav', 'against.sav', 'positive.sav', 'negative.sav','factual.sav', 'emotional.sav' )

def test():
    train = pd.read_csv('negative.csv')
    print("Training Set:"% train.columns, train.shape, len(train))
    pipeline_ridge = Pipeline([
                               ('vect', CountVectorizer()),
                               ('tfidf',  TfidfTransformer()),
                               ('rd', LinearRegression()),
                               ])
    scores = cross_val_score(pipeline_ridge, train['tweet'], train['stance'], scoring = 'neg_mean_squared_error', cv=5)
    rmse_s = []
    for score in scores:
        rmse_s.append(math.sqrt(abs(score)))
    rmse = np.mean(rmse_s)
    print(rmse)
    model = pipeline_ridge.fit(train['tweet'], train['stance'])
    filename = 'linearRegression_negative.sav'
    joblib.dump(model, filename)

def main():
    for i in range(6):
        file = files[i]
        train = pd.read_csv(file)
        print("Training Set:"% train.columns, train.shape, len(train))
        pipeline_ridge = Pipeline([
                                 ('vect', CountVectorizer()),
                                 ('tfidf',  TfidfTransformer()),
                                 ('rd', Ridge()),
                                 ])
        pipeline_linearRegression = Pipeline([
                                 ('vect', CountVectorizer()),
                                 ('tfidf',  TfidfTransformer()),
                                 ('rd', LinearRegression()),
                                 ])
        pipeline_lasso = Pipeline([
                                 ('vect', CountVectorizer()),
                                 ('tfidf',  TfidfTransformer()),
                                 ('rd', Lasso()),
                                 ])
        idx = -1
        min_rmse = 999;
        for j in range(2):
            if j == 0:
                scores = cross_val_score(pipeline_ridge, train['tweet'], train['stance'], scoring = 'neg_mean_squared_error', cv=5)
            elif j == 1:
                scores = cross_val_score(pipeline_linearRegression, train['tweet'], train['stance'], scoring = 'neg_mean_squared_error', cv=5)
            else:
                scores = cross_val_score(pipeline_lasso, train['tweet'], train['stance'], scoring = 'neg_mean_squared_error', cv=5)
            rmse_s = []
            for score in scores:
                rmse_s.append(math.sqrt(abs(score)))
            rmse = np.mean(rmse_s)
#            print(rmse)
            if rmse < min_rmse:
                min_rmse = rmse
                idx = j
        print(min_rmse)
        if idx == 0:
            model = pipeline_ridge.fit(train['tweet'], train['stance'])
            filename = 'ridge_'+ml_models[i]
            joblib.dump(model, filename)
        elif idx == 1:
            model = pipeline_linearRegression.fit(train['tweet'], train['stance'])
            filename = 'linearRegression_'+ml_models[i]
            joblib.dump(model, filename)
        else:
            model = pipeline_lasso.fit(train['tweet'], train['stance'])
            filename = 'lasso_'+ml_models[i]
            joblib.dump(model, filename)

if __name__ == '__main__':
    main()
#    test()
