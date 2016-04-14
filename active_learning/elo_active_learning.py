from constants import CONST

import util
import numpy as np

from sklearn import ensemble
from sklearn.metrics import mean_squared_error

from _collections import defaultdict

__author__ = 'hurshprasad'


class ELO_ACTIVE_LEARNING(object):
    def __init__(self):
        self.L = None  # set base L sample set from main, 2K, 4K, 8K
        self.L_labels = None

        self.M_selected_examples = 0

        self.U = util.load(CONST.DATASET_PATH + CONST.ACTIVETRAING_U_PATH)
        self.Q = util.load(CONST.DATASET_PATH + CONST.ACTIVELEARNING_Q_PATH)
        self.Y = util.load(CONST.DATASET_PATH + CONST.ACTIVELEARNING_Y_PATH)

        self.selected_Q = None
        self.selected_D = None

        self.NDCG = None

        # Gradient Boosting Decision Tree - Regression
        # Parameters, hyper-params pre-selected in this case
        self.gbdt_params = {'n_estimators': 500,
                            'max_depth': 4,
                            'min_samples_split': 5,
                            'learning_rate': 0.01,
                            'loss': 'ls'}

        self.learner_estimator = None

    # Algorithm 1
    def query_level_elo(self, basename):

        """
        PSEUDO-CODE ~ Require: Labeled set L, unlabeled set U
        for i=1,. . . ,N
            do N =size of the ensemble Subsample L
            and learn a relevance function
            s^i_j <-- score predicted by that function on the j-th document in U.
        end for
        """

        s_i_j = np.empty([self.U.shape[0], len(CONST.LEARNERS_N)])

        N = 0
        mse = 0.0
        for estimator in CONST.LEARNERS_N:
            self.gbdt_params.__setitem__('n_estimators', estimator)
            learner = ensemble.GradientBoostingRegressor(**self.gbdt_params)
            learner.fit(self.L, self.L_labels[:, 0].astype(float))

            # get prediction
            Y_pred = learner.predict(self.U)
            s_i_j[:, N] = Y_pred  # append score predictions to matrix
            mse += mean_squared_error(self.Y[:, 0].astype(float), Y_pred)
            N += 1

        print base_labeled, "MSE: %.4f" % (mse / N),

        unique_queries = np.unique(self.Y[:, 1], return_index=False)
        query_scores = defaultdict(float)

        for query in unique_queries:
            print "\rScoring query", query, "in L",
            # get labeled x, y for the learner
            score = self.get_el_by_query(query, s_i_j)
            # Learner gradient boosting decision tree (sklearn)
            query_scores[query] = score

        query_el_scores = sorted(query_scores, key=query_scores.get, reverse=True)

        return query_el_scores

    @staticmethod
    def calculate_matrix_bdcg(doc_gain_scores_j):

        if len(doc_gain_scores_j.shape) > 1:
            d_i = np.empty(doc_gain_scores_j.shape)
            for column in xrange(0, doc_gain_scores_j.shape[1]):
                g_s_j = doc_gain_scores_j[:, column]
                d_i[:, column] = ELO_ACTIVE_LEARNING.calculate_best_dcg_vector(g_s_j)

            return np.sum(d_i, axis=1)
        else:
            return ELO_ACTIVE_LEARNING.calculate_best_dcg_vector(doc_gain_scores_j)

    @staticmethod
    def calculate_best_dcg_vector(g_s_j):
        pi = np.add(np.arange(len(g_s_j)), 1)
        g_s_j.sort()  # sort
        g_s_j = g_s_j[::-1]  # sort decreasing order

        denom = np.log(np.add(1, pi))
        bdcg = np.divide(g_s_j, denom)

        return bdcg

    def get_el_by_query(self, query, s_i_j):

        # get index's of all documents
        # in Y (where index and rel score is) by query
        Xq_indicies = np.where(self.Y == query)

        G_s = np.subtract(np.power(s_i_j, 2), 1)  # gain vector for U

        s_i_j_I = G_s[Xq_indicies[0]]  # score for all documents in query

        t_j = np.mean(G_s, axis=1)

        d_i = ELO_ACTIVE_LEARNING.calculate_matrix_bdcg(s_i_j_I)

        d = ELO_ACTIVE_LEARNING.calculate_matrix_bdcg(t_j[Xq_indicies[0]])

        return np.mean(d_i, axis=0) - np.sum(d, axis=0)

    def document_level_elo_algorithm(self, query):

        # get j documents from L associated to query Q
        Xq_indicies = np.where(self.Y == query)

        # get score predictions for
        s_i_j = np.empty([Xq_indicies[0].shape[0], len(CONST.LEARNERS_N)])
        N = 0
        mse = 0.0
        for estimator in CONST.LEARNERS_N:
            self.gbdt_params.__setitem__('n_estimators', estimator)
            learner = ensemble.GradientBoostingRegressor(**self.gbdt_params)
            learner.fit(self.L, self.L_labels[:, 0].astype(float))

            # in document level only predict on Xq documents
            Y_pred = learner.predict(self.U)
            s_i_j[:, N] = Y_pred[Xq_indicies[0]]  # append score predictions to matrix

            Y_labels = self.Y[Xq_indicies[0]]
            mse += mean_squared_error(Y_labels[:, 0].astype(float), Y_pred[Xq_indicies[0]])
            N += 1

        print base_labeled, "MSE: %.4f" % (mse / N), "",

        # initialize expected loss for document j
        el_j = np.zeros(Xq_indicies[0].shape[0])

        # keep track of index, because we perform replacement

        for j in xrange(0, len(el_j)):
            print "\rScoring document %d" % j, "for query", query, "in U",
            t_k = np.delete(s_i_j, j, 0)

            d_p = ELO_ACTIVE_LEARNING.calculate_matrix_bdcg(t_k)

            g_j = np.take(s_i_j, j, 0)

            # get rank of j
            ranked = np.argsort(np.mean(s_i_j, 1))
            ranked = ranked[::-1]
            pi = np.argwhere(ranked == j)
            pi = pi[0][0] + 1

            # g_k = np.subtract(np.power(s_i_j, 2), 1)  # gain vector for U

            bgcd_j = np.divide(np.mean(g_j, 0), np.log(1 + pi))

            el_j[j] = el_j[j] + np.mean(d_p) - bgcd_j

        return el_j, Xq_indicies[0]

    def load_base_labels(self, base_path):
        self.L = util.load(CONST.DATASET_PATH + base_path)
        self.L_labels = util.load(CONST.DATASET_PATH + base_path + '_y')

    def perform_elo_active_learning(self, base_labeled):

        self.load_base_labels(base_labeled)

        # Algorithm 1 - Query Level
        top_queries = self.query_level_elo(base_labeled)

        dcg = []

        sample_threshold = 9

        # Algorithm 2 - Document Level
        for query in top_queries:

            el_j, d_j = self.document_level_elo_algorithm(query)

            count = self.M_selected_examples
            save_dcg_for_plot = False
            if self.M_selected_examples < CONST.SATURATION_MAX:
                sorted_el_j = np.argsort(el_j)
                sorted_el_j = sorted_el_j[::-1]

                range_size = sorted_el_j.shape[0] if sorted_el_j.shape[0] < CONST.M \
                    else CONST.M

                # free for all query examples for L suck it U
                for index in range(0, range_size):
                    print "\rTransfering document", self.M_selected_examples + 1, "for query", query, "from U to L",
                    transfer = np.take(self.U, d_j[sorted_el_j[index]], 0)
                    self.L = np.vstack((self.L, transfer))
                    self.L_labels = np.vstack((self.L_labels, self.Y[d_j[sorted_el_j[index]]]))
                    #self.U = np.delete(self.U, (d_j[sorted_el_j[index]]), 0)
                    self.M_selected_examples += 1
                    if self.M_selected_examples > np.power(2, sample_threshold):
                        save_dcg_for_plot = True

            # Don't care after certain saturation
            # Time to plot
            if sample_threshold > 16:
                break

            if save_dcg_for_plot:
                sample_threshold += 1
                print "Calculate DCG@10", self.M_selected_examples, sample_threshold
                # DCG POINT! Take

        return dcg  # return NDCG score


if __name__ == '__main__':

    # Load Training Data
    base_labeled_examples = {CONST.BASE2K_PATH}  # ,
    # CONST.BASE4K_PATH,
    # CONST.BASE8K_PATH}

    active_learn_elo = ELO_ACTIVE_LEARNING()

    ndcg_10 = []

    for base_labeled in base_labeled_examples:
        print '\nLoading Base L', base_labeled, '\n'
        # Now select queries
        active_learn_elo.perform_elo_active_learning(base_labeled)

    print "\nCompleted ELO Active Learning..."
