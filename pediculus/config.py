#The stock ticker
stock = "AAPL"
#How far the data is collected from, this is from a random interval inside the 16 year span. (Measured in DAYS)
LOOKBACK = 60
BATCH_SIZE = 32
EPOCHS = 100
BATCHES_PER_EPOCH = 100

#Model
filters = 32
kernalSize = 3
activation = "relu"
poolSize = 2
dropout = 0.2
dense = 1

