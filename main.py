from utils import read_data
from crf import CRF
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
import scipy
import pickle
import numpy as np


def random_search(labels, X_train, y_train, crf):
    params_space = {
        'c1': np.random.exponential(scale=0.3, size=10),
        'c2': np.random.exponential(scale=0.3, size=10)
    }

    f1_score = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)

    rs = RandomizedSearchCV(crf, params_space,
                            cv=5,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_score,
                            refit='accuracy',
                            return_train_score=True
                            )

    best_model = rs.fit(X_train, y_train)

    return best_model


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == '__main__':
    data_train = read_data('data/preprocessed/train.txt')
    data_test = read_data('data/preprocessed/test.txt')
    X_train = [[(token, pos, chunk) for token, pos, chunk, _ in sentence] for sentence in data_train]
    y_train = [[label for _, _, _, label in sentence] for sentence in data_train]
    X_test = [[(token, pos, chunk) for token, pos, chunk, _ in sentence] for sentence in data_test]
    y_test = [[label for _, _, _, label in sentence] for sentence in data_test]
    labels = pickle.load(open('labels.pkl', 'rb'))
    labels.remove('O')

    crf = CRF(is_using_pos_chunk=True, c1=0.04, c2=0.08)
    # best_model = random_search(labels, X_train, y_train, crf)
    # print('Best c1:', best_model.best_estimator_.get_params()['c1'])
    # print('Best c2:', best_model.best_estimator_.get_params()['c2'])
    # report(best_model.cv_results_)

    # best c1,c2: 0.026,0.037

    crf.fit(X_train, y_train)
    with open('data/model/crf_model_pos_chunk.pkl', mode='wb') as f:
        pickle.dump(crf, f)
        f.close()

    y_pred = crf.predict(X_test)

    with open('data/output/output_pos_chunk.txt', mode='w', encoding='utf8', newline='\n') as f:
        for i, sentence in enumerate(X_test[:-1]):
            tags = y_pred[i]
            y_test_i = y_test[i]
            for (token, pos, chunk), true_tag, tag in zip(sentence, y_test_i, tags):
                f.write(f"{token}\t{pos}\t{true_tag}\t{tag}\n")
            f.write('\n')

        X_test_last = X_test[-1]
        y_pred_last = y_pred[-1]
        y_test_last = y_test[-1]

        for (token, pos, chunk), true_tag, tag in zip(X_test_last[:-1], y_test_last[:-1], y_pred_last[:-1]):
            f.write(f"{token}\t{pos}\t{true_tag}\t{tag}\n")
        f.write(f"{X_test_last[-1][0]}\t{X_test_last[-1][1]}\t{y_test_last[-1]}\t{y_pred_last[-1]}")
        f.close()

# & Precision &  Recall  & F$_{\beta=1} \\\hline
# LOC     &   85.73\% &  87.16\% &  86.44 \\
# MISC    &   92.50\% &  75.51\% &  83.15 \\
# ORG     &   71.17\% &  42.34\% &  53.09 \\
# PER     &   91.43\% &  84.93\% &  88.06 \\\hline
# Overall &   87.42\% &  81.91\% &  84.58 \\\hline
