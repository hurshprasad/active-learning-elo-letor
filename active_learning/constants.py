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
    def BASE2K_PATH():
        return "pre_processed/base2k"

    @constant
    def BASE4K_PATH():
        return "pre_processed/base4k"

    @constant
    def BASE8K_PATH():
        return "pre_processed/base8k"

    @constant
    def ACTIVETRAING_PATH():
        return "pre_processed/AL"

    @constant
    def NUM_TRAINING_EXAMPLES():
        return 20000

    @constant
    def NUM_ACTIVETRAINING_EXAMPLES():
        return 2000

    @constant
    def RANK_INDEX_BM25():
        return 0

CONST = _Const()
