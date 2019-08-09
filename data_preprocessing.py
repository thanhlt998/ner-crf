import re


def preprocess(fn, output_fn):
    with open(fn, mode='r', encoding='utf8') as f_read, open(output_fn, mode='w', encoding='utf8') as f_write:
        for line in f_read:
            f_write.write(re.sub(r' ', '_', line))
        f_read.close()
        f_write.close()


if __name__ == '__main__':
    preprocess('data/raw/train.txt', 'data/preprocessed/train.txt')
    preprocess('data/raw/test.txt', 'data/preprocessed/test.txt')
