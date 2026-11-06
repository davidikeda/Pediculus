import os
import numpy as np
import pandas as pd

from dataPipeline import prep, createSubset
from model import build, train, calculate_FP_rate
from config import STOCKS, DATASET_SIZES, TRAILS_PER_SIZE, LOOKBACK

def evaluate(model, Xtest, ytest):
    predictions = (model.predict(Xtest, verbose=0).flatten() >= 0.5).astype(int)
    accy = (predictions == ytest).mean()
    fpr = calculate_FP_rate(ytest, predictions)
    return accy, fpr

def runSingle(ticker, fraction, seed):
    np.random.seed(seed)

    Xtrain, ytrain, Xval, yval, Xtest, ytest = prep(ticker)

    XtrainSub, ytrainSub = createSubset(Xtrain, ytrain, fraction)

    model = build(iShape =(LOOKBACK, XtrainSub.shape[2]))

    train(model, XtrainSub, ytrainSub, Xval, yval)

    accy, fpr = evaluate(model, Xtest, ytest)

    return {
        "ticker": ticker,
        "fraction": fraction,
        "seed": seed,
        "accuracy": accy,
        "false_positive_rate": fpr
    }

def run():
    results = []

    for i in DATASET_SIZES:
        for trial in range(TRAILS_PER_SIZE):
            seed = 1000 + trial + int(i * 100)

            for ticker in STOCKS:
                result = runSingle(ticker, i, seed)
                results.append(result)

        df = pd.DataFrame(results)
    return df