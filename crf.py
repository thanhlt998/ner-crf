import sklearn_crfsuite
from sklearn.base import BaseEstimator, TransformerMixin

from utils import get_features


class CRF(BaseEstimator, TransformerMixin):
    def __init__(self, is_using_pos_chunk=False, algorithm='lbfgs', c1=0.06, c2=0.1, max_iterations=100, all_possible_transitions=True):
        self.is_using_pos_chunk = is_using_pos_chunk
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2, max_iterations=max_iterations,
                                          all_possible_transitions=all_possible_transitions)

    def fit(self, X, y):
        features = [get_features(x, is_using_pos_chunk=self.is_using_pos_chunk) for x in X]
        self.model.fit(features, y)

    def predict(self, X):
        features = [get_features(x, is_using_pos_chunk=self.is_using_pos_chunk) for x in X]
        tags = self.model.predict(features)
        return tags
