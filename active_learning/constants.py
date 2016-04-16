# constants

__author__ = 'hurshprasad'


def constant(f):
    def fset(self, value):
        raise TypeError

    def fget(self):
        return f()

    return property(fget, fset)


class _Const(object):


    @constant
    def DATASET_PATH():
        return "../data/MQ2016/active_learning/"

    @constant
    def TRAINING():
        return "train.txt"

    @constant
    def VALIDATION():
        return "vali.txt"

    @constant
    def TESTING():
        return "test.txt"

    @constant
    def TEST_PATH():
        return "pre_processed/test_data"

    @constant
    def TEST_PATH_LABELS():
        return "pre_processed/test_data_labels"

    @constant
    def TEST_PATH_Q():
        return "pre_processed/test_data_q"

    @constant
    def TEST_PATH_DOCS():
        return "pre_processed/test_data_docs"

    @constant
    def BASE2K_PATH():
        return "pre_processed/base2k"

    @constant
    def BASE4K_PATH():
        return "pre_processed/base4k"

    @constant
    def BASE8K_PATH():
        return "pre_processed/base8k"

    @constant
    def ACTIVETRAING_U_PATH():
        return "pre_processed/AL"

    @constant
    def ACTIVELEARNING_Y_PATH():
        return "pre_processed/al_y"

    @constant
    def ACTIVELEARNING_Q_PATH():
        return "pre_processed/al_q"

    @constant
    def ACTIVELEARNING_DOCNAMES_PATH():
        return "pre_processed/al_doc"

    # 208869 Total Examples in data/MQ2016/active_learning/train.txt
    #
    # 160,000 for AL ~ U
    #   2,000 BASE 2K
    #   4,000 BASE 4K
    #   8,000 BASE 8K
    # ---------------
    # 174,000 TOTAL
    @constant
    def NUM_TRAINING_EXAMPLES():
        return 174000

    @constant
    def NUM_ACTIVETRAINING_EXAMPLES():
        return 160000

    @constant
    def SATURATION_MAX():
        return 65536

    @constant
    def SATURATION_EPSILON():
        return 0.5

    @constant
    def RANK_INDEX_BM25():
        return 0  # https://github.com/verayan/LETOR/blob/master/Readme_TREC_dataset.pdf

    @constant
    def LEARNERS_N():
        return {100,
                150,
                200,
                250}  # estimator hyper_param for gradient boosting decision tree

    @constant
    def M():
        return 500  # Number of top examples to move from U to L, if possible

    @constant
    def SAMPLE_TICKS():
        return 16  # Number of 2^X where we sample DCG@10 starting from 9

    @constant
    def BASE_TESTS():
        #return {CONST.BASE2K_PATH, CONST.BASE4K_PATH, CONST.BASE8K_PATH}
        return {CONST.BASE2K_PATH}

CONST = _Const()
