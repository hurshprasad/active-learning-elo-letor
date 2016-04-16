from active_learning.constants import CONST
import util
import math
import numpy as np

__author__ = 'hurshprasad'

# normalize each feature set
def normalize(row):
    s = math.sqrt(reduce(lambda x,y: x + y*y,row))
    assert s != 0.0, "S can not be 0"
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
            tempy = []

            for value in values[0:2]:
                if value.__contains__(":"):
                    tempy.append(value.split(":")[1])
                else:
                    tempy.append(value)

            Y.append(tempy)
            docs.append(values[50])

    tempy = np.asarray(Y)
    Q = np.unique(tempy[:, 1], return_index=False)

    return np.array(X), np.array(Y), np.array(Q), np.array(docs)

def read_test_data(file):

    # assume if one is saved they all are
    if util.check_file_exists(CONST.DATASET_PATH + CONST.TEST_PATH):
        T_Data = util.load(CONST.DATASET_PATH + CONST.TEST_PATH)
        T_Labels = util.load(CONST.DATASET_PATH + CONST.TEST_PATH_LABELS)
        T_Queries = util.load(CONST.DATASET_PATH + CONST.TEST_PATH_Q)
        T_Docs = util.load(CONST.DATASET_PATH + CONST.TEST_PATH_DOCS)

    else:
        T_Data, T_Labels, T_Queries, T_Docs = read_train_data(file)

        util.save_pickle(CONST.DATASET_PATH + CONST.TEST_PATH, T_Data)
        util.save_pickle(CONST.DATASET_PATH + CONST.TEST_PATH_LABELS, T_Labels)
        util.save_pickle(CONST.DATASET_PATH + CONST.TEST_PATH_Q, T_Queries)
        util.save_pickle(CONST.DATASET_PATH + CONST.TEST_PATH_DOCS, T_Docs)

    return T_Data, T_Labels, T_Queries, T_Docs

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

    # get labels for L base set
    two_y = Y[two]
    four_y = Y[four]
    eight_y = Y[eight]

    # base set L examples
    two = X[two]
    four = X[four]
    eight = X[eight]

    # save BASE set
    util.save_pickle(CONST.DATASET_PATH + CONST.BASE2K_PATH, two)
    util.save_pickle(CONST.DATASET_PATH + CONST.BASE4K_PATH, four)
    util.save_pickle(CONST.DATASET_PATH + CONST.BASE8K_PATH, eight)

    # save BASE Y set
    util.save_pickle(CONST.DATASET_PATH + CONST.BASE2K_PATH + '_y', two_y)
    util.save_pickle(CONST.DATASET_PATH + CONST.BASE4K_PATH + '_y', four_y)
    util.save_pickle(CONST.DATASET_PATH + CONST.BASE8K_PATH + '_y', eight_y)

    # U set
    Y = Y[AL]
    D = D[AL]

    AL = X[AL]

    # save AL, U set
    util.save_pickle(CONST.DATASET_PATH + CONST.ACTIVETRAING_U_PATH, AL)

    # SAVE Y
    util.save_pickle(CONST.DATASET_PATH + CONST.ACTIVELEARNING_Y_PATH, Y)

    # SAVE Q
    util.save_pickle(CONST.DATASET_PATH + CONST.ACTIVELEARNING_Q_PATH, Q)

    # SAVE DOCS
    util.save_pickle(CONST.DATASET_PATH + CONST.ACTIVELEARNING_DOCNAMES_PATH, D)

if __name__ == "__main__":
    main()
