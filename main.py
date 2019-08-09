from utils import read_data
from crf import CRF
import pickle

if __name__ == '__main__':
    data_train = read_data('data/preprocessed/train.txt')
    data_test = read_data('data/preprocessed/test.txt')
    X_train = [[(token, pos, chunk) for token, pos, chunk, _ in sentence] for sentence in data_train]
    y_train = [[label for _, _, _, label in sentence] for sentence in data_train]
    X_test = [[(token, pos, chunk) for token, pos, chunk, _ in sentence] for sentence in data_test]
    y_test = [[label for _, _, _, label in sentence] for sentence in data_test]

    crf = CRF(is_using_pos_chunk=False)
    crf.fit(X_train, y_train)
    with open('data/model/crf_model_no_pos_chunk.pkl', mode='wb') as f:
        pickle.dump(crf, f)
        f.close()

    y_pred = crf.predict(X_test)

    with open('data/output/output_pos_chunk.txt', mode='w', encoding='utf8') as f:
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
