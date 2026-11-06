import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from config import (
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
    PREDICTION_HORIZON,
    FEATURE_COLUMNS,
    LOOKBACK,
    RAW,
)


def normalizeColumns(df):
    """Convert yfinance columns—including MultiIndex columns—to standard names."""
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        standard_names = {
            "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"
        }
        new_columns = []

        for column in df.columns:
            parts = [str(part) for part in column]
            name = next(
                (part for part in parts if part in standard_names),
                parts[0],
            )
            new_columns.append(name)

        df.columns = new_columns
    else:
        df.columns = [str(column) for column in df.columns]

    return df


def downloadStockData(ticker):
    """Load cached stock data or download it from Yahoo Finance."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{ticker}.csv"

    if path.exists():
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    else:
        df = yf.download(
            ticker,
            start=TRAIN_START,
            end="2026-01-10",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if df.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        df = normalizeColumns(df).reset_index()
        df.to_csv(path, index=False)

    df = normalizeColumns(df)

    if "Date" not in df.columns:
        df = df.reset_index()
        df = normalizeColumns(df)

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    required = {"Date", "Adj Close", *FEATURE_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{ticker} data is missing columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def buildLabelColumn(df, horizon=PREDICTION_HORIZON):
    """
    Label each date as 1 if adjusted close is higher after `horizon`
    trading days, otherwise 0.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    df["Future_Date"] = df["Date"].shift(-horizon)
    df["Future_Adj_Close"] = df["Adj Close"].shift(-horizon)

    df = df.dropna(subset=["Future_Date", "Future_Adj_Close"]).copy()
    df["Target"] = (
        df["Future_Adj_Close"] > df["Adj Close"]
    ).astype(int)

    return df.reset_index(drop=True)


def createTimeSplit(df):
    """Create chronological train, validation, and test splits."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Future_Date"] = pd.to_datetime(df["Future_Date"])

    train_start = pd.Timestamp(TRAIN_START)
    train_end = pd.Timestamp(TRAIN_END)
    val_start = pd.Timestamp(VAL_START)
    val_end = pd.Timestamp(VAL_END)
    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    train_mask = (
        (df["Date"] >= train_start)
        & (df["Date"] <= train_end)
        & (df["Future_Date"] <= train_end)
    )

    val_mask = (
        (df["Date"] >= val_start)
        & (df["Date"] <= val_end)
        & (df["Future_Date"] <= val_end)
    )

    test_mask = (
        (df["Date"] >= test_start)
        & (df["Date"] <= test_end)
    )

    return (
        df.loc[train_mask].copy(),
        df.loc[val_mask].copy(),
        df.loc[test_mask].copy(),
    )


def makeFeatureMatrix(df):
    """Select the configured numerical input features."""
    return df[FEATURE_COLUMNS].astype(float)


def createWindow(X, y, lookback=LOOKBACK):
    """Convert rows into sliding windows with the label for each window's last date."""
    X_windows = []
    y_windows = []

    for i in range(len(X) - lookback + 1):
        X_windows.append(X[i : i + lookback])
        y_windows.append(y[i + lookback - 1])

    if not X_windows:
        raise ValueError(
            f"Not enough data to create windows: got {len(X)} rows, "
            f"need at least {lookback}."
        )

    return np.asarray(X_windows), np.asarray(y_windows, dtype=int)


def prep(ticker):
    """Prepare train, validation, and test arrays for one ticker."""
    df = downloadStockData(ticker)
    df = buildLabelColumn(df)

    train_df, val_df, test_df = createTimeSplit(df)

    for name, split_df in (
        ("training", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        if len(split_df) < LOOKBACK:
            raise ValueError(
                f"Not enough {name} rows for {ticker}: "
                f"got {len(split_df)}, need at least {LOOKBACK}."
            )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(makeFeatureMatrix(train_df))
    X_val_scaled = scaler.transform(makeFeatureMatrix(val_df))
    X_test_scaled = scaler.transform(makeFeatureMatrix(test_df))

    y_train = train_df["Target"].to_numpy()
    y_val = val_df["Target"].to_numpy()
    y_test = test_df["Target"].to_numpy()

    X_train, y_train = createWindow(X_train_scaled, y_train)
    X_val, y_val = createWindow(X_val_scaled, y_val)
    X_test, y_test = createWindow(X_test_scaled, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def createSubset(Xtrain, ytrain, fraction):
    """Take the first fraction of training examples, preserving chronology."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")

    n = max(1, int(len(Xtrain) * fraction))
    return Xtrain[:n], ytrain[:n]