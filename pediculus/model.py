import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam


from config import LOOKBACK, MODEL_FILTERS, KERNEL_SIZE, DROPOUT, LEARN_RATE, EPOCHS, BATCH_SIZE

def build(iShape):
    model = tf.keras.Sequential([
        Conv1D(filters=MODEL_FILTERS, kernel_size=KERNEL_SIZE, activation="relu", input_shape=iShape),
        MaxPooling1D(pool_size=2),
        Dropout(DROPOUT),
        Conv1D(filters=MODEL_FILTERS // 2, kernel_size=KERNEL_SIZE, activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(DROPOUT),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARN_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train(model, Xtrain, ytrain, Xval, yval):
    history = model.fit(
        Xtrain,
        ytrain,
        validation_data=(Xval, yval),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    return history

def calculate_FP_rate(ytrue, ypred):
    """
    False Positive rate (FP Rate) = FP / (FP + TN)
    positive class = "up"
    """

    ytrue = ytrue.astype(int)
    ypred = ypred.astype(int)

    tp = ((ytrue == 1) & (ypred == 1)).sum()
    tn = ((ytrue == 0) & (ypred == 0)).sum()
    fp = ((ytrue == 0) & (ypred == 1)).sum()
    fn = ((ytrue == 1) & (ypred == 0)).sum()

    if (fp + tn) == 0:
        return 0.0

    return fp / (fp + tn)