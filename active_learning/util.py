# util
import cPickle
import os

__author__ = 'hurshprasad'


def save_pickle(file, data):
    fp = open(file + '.pickle', 'w')
    cPickle.dump(data, fp, protocol=2)
    fp.close()


def load(file):
    assert os.path.isfile(file + '.pickle'), "file doesn't exist"
    fp = open(file + '.pickle', 'rb')
    data = cPickle.load(fp)
    fp.close()
    return data


def check_file_exists(file):
    if os.path.exists(file + '.pickle'):
        return True

    return False
