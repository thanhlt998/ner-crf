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
    max_words_per_line = 15
    for sentence, tag in zip(sentences, tags):
        formated_sentence = []
        formated_tag = []
        for (token,), tag_ in zip(sentence, tag):
            max_len = max(len(token), len(tag_)) + 4
            formated_sentence.append(fill(token, max_len))
            formated_tag.append(fill(tag_, max_len))

        no_lines = len(formated_sentence) // max_words_per_line + (
            0 if len(formated_sentence) % max_words_per_line == 0 else 1)

        for i in range(no_lines):
            print(' '.join(formated_sentence[max_words_per_line * i: max_words_per_line * (i + 1)]))
            print(' '.join(formated_tag[max_words_per_line * i: max_words_per_line * (i + 1)]))
            print('\n')


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
