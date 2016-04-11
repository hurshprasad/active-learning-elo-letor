from active_learning.constants import CONST
import util
import math
import numpy as np

__author__ = 'hurshprasad'

# normalize each feature set
def normalize(row):
    s = math.sqrt(reduce(lambda x,y: x + y*y,row))
    if s == 0.0:
        print "wtf"
    return map(lambda x : x/s, row)

# read training data
# return data matrix and class label
# for each query
def read_train_data(file):
    X = []
    Y = []
    docs = []

    with open(file, "r") as f:
        rows = f.readlines()
    for row in rows[0:CONST.NUM_TRAINING_EXAMPLES]:
        values = row.split(" ")

        temp = []
        for value in values[2:-9]:
                temp.append(float(value.split(":")[1]))

        if sum(temp) > 1.0:
            X.append(normalize(temp))
            tempY = []

            for value in values[0:2]:
                if value.__contains__(":"):
                    tempY.append(value.split(":")[1])
                else:
                    tempY.append(value)

            Y.append(tempY)
            docs.append(values[50])

    Y = np.asarray(Y)
    Q = np.unique(Y[:, 1], return_index=False)

    return np.array(X), np.array(Y), np.array(Q), np.array(docs)

def read_test_data(file):
    return None

def main():
    # read the configuration file
    #config = args.data

    train_file = CONST.DATASET_PATH + CONST.TRAINING

    # training set, Labels, Queries, Document Names
    X, Y, Q, D = read_train_data(train_file)

    print "Loaded Trainging Data ..."

    # get all the indicies
    indices = np.r_[0:X.shape[0]]

    ### SETUP BASE EXAMPLES ###

    # chose per N for L with replacement
    two = np.random.choice(indices, 2000)
    four = np.random.choice(indices, 4000)
    eight = np.random.choice(indices, 8000)

    # chose per N for U
    AL = np.random.choice(indices, CONST.NUM_ACTIVETRAINING_EXAMPLES)

    two = X[two]
    four = X[four]
    eight = X[eight]

    Y = Y[AL]
    D = D[AL]

    AL = X[AL]

    # save BASE set
    util.save_pickle(CONST.DATASET_PATH + CONST.BASE2K_PATH, two)
    util.save_pickle(CONST.DATASET_PATH + CONST.BASE4K_PATH, four)
    util.save_pickle(CONST.DATASET_PATH + CONST.BASE8K_PATH, eight)

    # save AL, U set
    util.save_pickle(CONST.DATASET_PATH + CONST.ACTIVETRAING_PATH, AL)

    # SAVE Y
    util.save_pickle(CONST.DATASET_PATH + CONST.ACTIVELEARNING_Y_PATH, Y)

    # SAVE Q
    util.save_pickle(CONST.DATASET_PATH + CONST.ACTIVELEARNING_Q_PATH, Q)

    # SAVE DOCS
    util.save_pickle(CONST.DATASET_PATH + CONST.ACTIVELEARNING_DOCNAMES_PATH, D)

if __name__ == "__main__":
    main()
