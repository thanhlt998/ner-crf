import pickle
from textwrap import wrap

from utils import get_sentences


def load_model(fn):
    with open(fn, mode='rb') as f:
        model = pickle.load(f)
        f.close()
    return model


def fill(text, width):
    text_len = len(text)
    head_space = (width - text_len) // 2
    tail_space = width - text_len - head_space
    return ''.join([' ' * head_space, text, ' ' * tail_space])


def print_result(sentences, tags):
    for sentence, tag in zip(sentences, tags):
        formated_sentence = []
        formated_tag = []
        for (token,), tag_ in zip(sentence, tag):
            max_len = max(len(token), len(tag_))
            formated_sentence.append(fill(token, max_len))
            formated_tag.append(fill(tag_, max_len))

        print(' '.join(formated_sentence))
        print(' '.join(formated_tag))


if __name__ == '__main__':
    model = load_model('data/model/crf_model_no_pos_chunk.pkl')
    is_stop = False
    while not is_stop:
        paragraph = input('Enter a paragraph: ')
        if paragraph == 'n' or paragraph == 'N':
            is_stop = True
        else:
            sentences = get_sentences(paragraph)
            tags = model.predict(sentences)
            print_result(sentences, tags)

