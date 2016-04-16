from constants import CONST

import util
import pre_processing

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

        self.U = None
        self.Q = None
        self.Y = None

        self.selected_Q = None
        self.selected_D = None

        self.T_Data, self.T_Labels, self.T_Queries, self.T_Docs = pre_processing.read_test_data(CONST.DATASET_PATH + CONST.TESTING)

        self.DCG = []

        # Gradient Boosting Decision Tree - Regression
        # Parameters, hyper-params pre-selected in this case
        self.gbdt_params = {'n_estimators': 500,
                            'max_depth': 4,
                            'min_samples_split': 5,
                            'learning_rate': 0.01,
                            'loss': 'ls'}

        self.learner_estimator = None

    """
    Get Expected Loss for all queries in U (Algorithm 1)
    """
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

    """
    Get Expected Loss per for a given query
    """
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

    """
    Get Expected Loss per document example for a given query (Algorithm 2)
    """
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
    """
    Load Data from pre-processed pickle files
    """
    def load_base_labels(self, base_path):
        self.L = None  # set base L sample set from main, 2K, 4K, 8K
        self.L_labels = None
        self.U = None
        self.Y = None
        self.Q = None

        self.M_selected_examples = 0

        self.DCG = []

        self.U = util.load(CONST.DATASET_PATH + CONST.ACTIVETRAING_U_PATH)
        self.Q = util.load(CONST.DATASET_PATH + CONST.ACTIVELEARNING_Q_PATH)
        self.Y = util.load(CONST.DATASET_PATH + CONST.ACTIVELEARNING_Y_PATH)

        self.L = util.load(CONST.DATASET_PATH + base_path)
        self.L_labels = util.load(CONST.DATASET_PATH + base_path + '_y')

    """
    predict on Test Set and calculate DCG@10
    """
    def get_test_dcg_10(self):
        self.gbdt_params.__setitem__('n_estimators', 100)
        learner = ensemble.GradientBoostingRegressor(**self.gbdt_params)
        learner.fit(self.L, self.L_labels[:, 0].astype(float))

        # in document level only predict on Xq documents
        T_pred = learner.predict(self.T_Data)

        #unique_queries = np.unique(self.T_Queries[:, 1], return_index=False)

        dcg10 = []
        for query in self.T_Queries:
            print "\rDCG@10 Scoring", "for query", query, " in Test Set DCG:", np.mean(dcg10) if len(dcg10) > 0 else 0,
            Tq_indicies = np.where(self.T_Labels == query)
            Tq_pred = T_pred[Tq_indicies[0]]  # append score predictions to matrix
            Tq_top = np.argsort(Tq_pred)
            Tq_top = Tq_top[::-1]   # reverse order

            if len(Tq_top) > 10:
                Tq_pred = Tq_pred[Tq_top[:10]]  # get top 10
            #Tq_p = Tq_pred[:10, :]  # get top 10

            dcg = Tq_pred[0]

            for i in xrange(1, len(Tq_pred)):
                dcg += Tq_pred[i]/np.log(i+1)

            dcg10.append(dcg)

        return np.mean(dcg10)

    """
    perform ELO Active Learning per base set
    """
    def perform_elo_active_learning(self, base_labeled):

        self.load_base_labels(base_labeled)

        # Algorithm 1 - Query Level
        top_queries = self.query_level_elo(base_labeled)

        sample_threshold = 9
        save_dcg_for_plot = False

        # Algorithm 2 - Document Level
        for query in top_queries:

            el_j, d_j = self.document_level_elo_algorithm(query)

            count = self.M_selected_examples
            if self.M_selected_examples < CONST.SATURATION_MAX:
                sorted_el_j = np.argsort(el_j)
                sorted_el_j = sorted_el_j[::-1]

                range_size = sorted_el_j.shape[0] if sorted_el_j.shape[0] < CONST.M \
                    else CONST.M

                # transfer selected examples from U to L
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
            if sample_threshold > CONST.SAMPLE_TICKS:
                break

            if save_dcg_for_plot:
                sample_threshold += 1
                save_dcg_for_plot = False
                self.DCG.append(self.get_test_dcg_10())
                print "\rCalculate DCG@10", self.M_selected_examples, sample_threshold, self.DCG[sample_threshold-10],


        return True  # return NDCG score


if __name__ == '__main__':

    # Load Training Data
    base_labeled_examples = CONST.BASE_TESTS

    active_learn_elo = ELO_ACTIVE_LEARNING()

    for base_labeled in base_labeled_examples:
        print '\nLoading Base L', base_labeled[14:], '\n'
        # Now select queries
        active_learn_elo.perform_elo_active_learning(base_labeled)
        print "\rCompleted ELO Active Learning", base_labeled, active_learn_elo.DCG, np.mean(active_learn_elo.DCG), '\n',
        print "-------------"
        #pylab.title(base_labeled[14:])
        #pylab.xlabel('Number of examples selected')
        #pylab.xticks(['2^9', '2^10', '2^11', '2^12', '2^13'])
        #pylab.ylabel("DCG@10")
        #pylab.plot(active_learn_elo.DCG)
        #pylab.plot(x, learning_curve_1, label='lambda=1')
        #pylab.legend(loc='upper right')
        #pylab.show()

    print "Done!"
