import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)

import subprocess
import os


# Config
from config import (
    stock,
    LOOKBACK,
    BATCH_SIZE,
    EPOCHS,
    BATCHES_PER_EPOCH,
    filters,
    kernalSize,
    activation,
    poolSize,
    dropout,
    dense
)


def checkIfWifiEnabled():
    try:
        output = subprocess.check_output(
            "netsh wlan show interfaces",
            shell=True
        ).decode("utf-8", errors="ignore")

        if "State" in output and "connected" in output.lower():
            for line in output.split("\n"):
                if "SSID" in line and "BSSID" not in line:
                    ssid = line.split(":")[1].strip()
                    print(f"Connected to Wi-Fi SSID: {ssid}")
                    return True

        print("Not connected to Wi-Fi.")
        return False

    except subprocess.CalledProcessError:
        print(
            "Could not check Wi-Fi status (Command failed). "
            "This causes the majority of the program to fail, "
            "as it needs to download the newest version of the stock data."
        )
        return False


def load(exp):
    filePath = os.path.join(
        "pediculus/datasets",
        stock + "Data.csv"
    )

    if not os.path.isfile(filePath):

        df = yf.download(
            stock,
            start="2010-01-01",
            end="2026-01-01"
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.to_csv(filePath)

    else:

        print(
            "The dataset: "
            + filePath
            + " exists; do you want to rewrite it? (Y)es, (N)o "
        )

        rewriteLoadFunction = str(input().upper())

        if (
            rewriteLoadFunction == "Y"
            or rewriteLoadFunction == "YES"
        ):

            df = yf.download(
                stock,
                start="2010-01-01",
                end="2026-01-01"
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_csv(filePath)

        elif (
            rewriteLoadFunction == "N"
            or rewriteLoadFunction == "NO"
        ):

            print("Ok! Using previous data.")
            df = pd.read_csv(filePath)

        else:

            print(
                "No action specified, defaulting to previous data :\\"
            )

            df = pd.read_csv(filePath)

    if exp != False:
        print(df)

    global features

    features = df[
        ["Close", "High", "Low", "Open", "Volume"]
    ].values


def randomBatch(data, lookback, batchSize):

    X = []
    Y = []

    MSI = len(data) - lookback - 1

    randomStarts = np.random.randint(
        0,
        MSI + 1,
        size=batchSize
    )

    for start in randomStarts:

        window = data[
            start:start + lookback
        ]

        target = data[
            start + lookback,
            2
        ]

        X.append(window)
        Y.append(target)

    return np.array(X), np.array(Y)


def useDaDataorsumting():

    scaler = StandardScaler()

    scaledFeatures = scaler.fit_transform(features)

    splitIDX = int(
        len(scaledFeatures) * 0.8
    )

    trainData = scaledFeatures[
        :splitIDX
    ]

    valData = scaledFeatures[
        splitIDX:
    ]

    X = []
    Y = []

    for i in range(
        len(valData) - LOOKBACK
    ):

        X.append(
            valData[
                i:i + LOOKBACK
            ]
        )

        Y.append(
            valData[
                i + LOOKBACK,
                2
            ]
        )

    return (
        trainData,
        np.array(X),
        np.array(Y)
    )


def buildModel():

    model = Sequential([

        Conv1D(
            filters=filters,
            kernel_size=kernalSize,
            activation=activation,
            input_shape=(LOOKBACK, 5)
        ),

        MaxPooling1D(
            pool_size=poolSize
        ),

        Dropout(
            dropout
        ),

        Conv1D(
            filters=int(filters / 2),
            kernel_size=kernalSize,
            activation=activation
        ),

        Flatten(),

        Dense(
            int(filters / 2),
            activation=activation
        ),

        Dropout(
            dropout
        ),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def Main():

    if checkIfWifiEnabled():

        load(True)

    else:

        print(
            "Offline, Attempting to load localized data"
        )

        try:

            load(False)

        except Exception as e:

            print(
                f"Failed to load local data: {e}"
            )

            exit()

    trainData, valX, valY = useDaDataorsumting()

    model = buildModel()

    print("Training")

    for epoch in range(EPOCHS):

        Eloss = 0

        for batch in range(
            BATCHES_PER_EPOCH
        ):


            batchX, batchY = randomBatch(
                trainData,
                lookback=LOOKBACK,
                batchSize=BATCH_SIZE
            )


            loss = model.train_on_batch(
                batchX,
                batchY
            )

            Eloss += loss[0]

        metrics = model.evaluate(
            valX,
            valY,
            verbose=0
        )

        trainLoss = (
            Eloss / BATCHES_PER_EPOCH
        )

        validationLoss = metrics[0]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"train mse loss: {trainLoss}"
        )

        print(
            f"Validation mse loss: {validationLoss}"
        )


Main()
