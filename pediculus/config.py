import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)

# Top 25 S&P 500 companies by index weight
STOCKS = [
    "AAPL",   # Apple Inc.
    "NVDA",   # NVIDIA Corporation
    "MSFT",   # Microsoft Corporation
    "AMZN",   # Amazon.com, Inc.
    "GOOGL",  # Alphabet Inc. (Class A)
    "AVGO",   # Broadcom Inc.
    "GOOG",   # Alphabet Inc. (Class C)
    "META",   # Meta Platforms, Inc.
    "LLY",    # Eli Lilly and Company
    "BRK-B",  # Berkshire Hathaway Inc. (Class B)
    "TSLA",   # Tesla, Inc.
    "JPM",    # JPMorgan Chase & Co.
    "MU",     # Micron Technology, Inc.
    "V",      # Visa Inc.
    "XOM",    # ExxonMobil Holdings Corporation
    "AMD",    # Advanced Micro Devices, Inc.
    "JNJ",    # Johnson & Johnson
    "WMT",    # Walmart Inc.
    "MA",     # Mastercard Incorporated
    "ABBV",   # AbbVie Inc.
    "COST",   # Costco Wholesale Corporation
    "CVX",    # Chevron Corporation
    "PG",     # The Procter & Gamble Company
    "CSCO",   # Cisco Systems, Inc.
    "KO"      # The Coca-Cola Company
]

#These are the dates that yfinance will pull info from, in YYYY/MM/DD format

TRAIN_START = "2015-01-01" # The markets were closed for new years day on this date, so the first day will be truncated
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

#How far the data is collected from, this is from a random interval inside the 16 year span. (Measured in DAYS)
LOOKBACK = 60
PREDICTION_HORIZON = 5
TARGET_COLUMN = "Adj Close"
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

DATASET_SIZES = [0.10, 0.25, 0.50, 0.75, 1.00]
TRAILS_PER_SIZE = 10

#Model hyperparams
BATCH_SIZE = 32
EPOCHS = 20
BATCHES_PER_EPOCH = 100
MODEL_FILTERS = 32
KERNEL_SIZE = 3
ACTIVATION = "relu"
POOLSIZE = 2
DROPOUT = 0.2
DENSE = 1
LEARN_RATE = 0.001

