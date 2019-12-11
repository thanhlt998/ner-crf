import re
from pyvi.ViTokenizer import ViTokenizer


def read_data(fn):
    sentences = []
    sentence = []
    with open(fn, mode='r', encoding='utf8') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                tokens = line.split()
                if len(tokens) == 4:
                    sentence.append(tuple(tokens))
            else:
                sentences.append(sentence)
                sentence = []
        if len(sentence) > 0:
            sentences.append(sentence)
        f.close()
    return sentences


def is_name(word):
    tokens = word.split('_')

    for token in tokens:
        if not token.istitle():
            return False

    return True


def is_mix_case(word):
    return len(word) > 2 and word[0].islower() and word[1].isupper()


def get_word_shape(word):
    word_shape = []
    for character in word:
        if character.isupper():
            word_shape.append('U')
        elif character.islower():
            word_shape.append('L')
        elif character.isdigit():
            word_shape.append('D')
        else:
            word_shape.append(character)
    return ''.join(word_shape)


def is_cap_with_period(word):
    return word[0].isupper() and word[-1] == '.'


def ends_with_digit(word):
    return word[-1].isdigit()


def contains_hyphen(word):
    return '-' in word


def is_date(word):
    return re.search(r"^([0-2]?[0-9]|30|31)[/-](0?[1-9]|10|11|12)([/-]\d{4})?$", word) is not None


def is_range(word):
    if re.match(r"^\d+-\d+$", word) is not None:
        nums = re.split(r'-', word)
        first_num = int(nums[0])
        second_num = int(nums[1])
        return first_num < second_num and second_num - first_num < 1000

    return False


def is_rate(word):
    if re.match(r"^\d+/\d+$", word) is not None:
        nums = re.split(r'/', word)
        first_num = int(nums[0])
        second_num = int(nums[1])
        return first_num < second_num
    return False


def is_month_year(word):
    return re.match(r"^(0?[1-9]|11|12)[/-]\d{4}$", word) is not None


def is_code(word):
    return word[0].isdigit() and word[-1].isupper()


def digit_and_comma(word):
    return re.search(r"^\d+,\d+$", word) is not None


def digit_and_period(word):
    return re.search(r"^\d+\.\d+$", word) is not None


def word_to_features(sentence, i, is_using_pos_chunk):
    word = sentence[i][0]

    features = {
        'w(0)': word,
        'w(0)[:1]': word[:1],
        'w(0)[:2]': word[:2],
        'w(0)[:3]': word[:3],
        'w(0)[:4]': word[:4],
        'w(0)[-1:]': word[-1:],
        'w(0)[-2:]': word[-2:],
        'w(0)[-3:]': word[-3:],
        'w(0)[-4:]': word[-4:],

        'word.islower': word.islower(),
        'word.lower': word.lower(),
        'isTitle': word.istitle(),
        'isNumber': word.isdigit(),
        'isUpper': word.isupper(),
        'isCapWithPeriod': is_cap_with_period(word),
        'endsWithDigit': ends_with_digit(word),
        'containsHyphen': contains_hyphen(word),
        'isDate': is_date(word) or is_month_year(word),
        'isCode': is_code(word),
        'isName': is_name(word),
        'isMixCase': is_mix_case(word),
        'd&comma': digit_and_comma(word),
        'd&period': digit_and_period(word),
        'wordShape': get_word_shape(word),

        'isRange': is_range(word),
        'isRate': is_rate(word)

    }

    if is_using_pos_chunk:
        pos = sentence[i][1]
        chunk = sentence[i][2]
        features.update({
            'pos': pos,
            'chunk': chunk,
        })

    if i > 0:
        previous_word = sentence[i - 1][0]

        features.update({
            'w(-1)': previous_word,
            'w(-1).lower': previous_word.lower(),
            'isTitle(-1)': previous_word.istitle(),
            'isNumber(-1)': previous_word.isdigit(),
            'isCapWithPeriod(-1)': is_cap_with_period(previous_word),
            'isName(-1)': is_name(previous_word),
            'wordShape(-1)': get_word_shape(previous_word),
            'w(-1)+w(0)': previous_word + ' ' + word,
        })

        if is_using_pos_chunk:
            pos = sentence[i][1]
            chunk = sentence[i][2]
            previous_pos = sentence[i - 1][1]
            previous_chunk = sentence[i - 1][2]
            features.update({
                'pos(-1)': previous_pos,
                'chunk(-1)': previous_chunk,
                'pos(-1)+pos(0)': previous_pos + ' ' + pos,
                'chunk(-1)+chunk(0)': previous_chunk + ' ' + chunk
            })
    else:
        features['BOS'] = True

    if i > 1:
        previous_word = sentence[i - 1][0]
        previous_2_word = sentence[i - 2][0]

        features.update({
            'w(-2)': previous_2_word,
            'w(-2)+w(-1)': previous_2_word + ' ' + previous_word,
            'isTitle(-2)': previous_2_word.istitle(),
            'isNumber(-2)': previous_2_word.isdigit()
        })

        if is_using_pos_chunk:
            previous_pos = sentence[i - 1][1]
            previous_2_pos = sentence[i - 2][1]
            previous_chunk = sentence[i - 1][2]
            previous_2_chunk = sentence[i - 2][2]
            features.update({
                'pos(-2)': previous_2_pos,
                'chunk(-2)': previous_2_chunk,
                'pos(-2)+pos(-1)': previous_2_pos + ' ' + previous_pos,
                'chunk(-2)+chunk(-1)': previous_2_chunk + ' ' + previous_chunk
            })

    if i < len(sentence) - 1:
        next_word = sentence[i + 1][0]

        features.update({
            'w(+1)': next_word,
            'w(+1).lower': next_word.lower(),
            'isTitle(+1)': next_word.istitle(),
            'isNumber(+1)': next_word.isdigit(),
            'isCapWithPeriod(+1)': is_cap_with_period(next_word),
            'isName(+1)': is_name(next_word),
            'wordShape(+1)': get_word_shape(next_word),
            'w(0)+w(+1)': word + ' ' + next_word,
        })
        if is_using_pos_chunk:
            pos = sentence[i][1]
            chunk = sentence[i][2]
            next_pos = sentence[i + 1][1]
            next_chunk = sentence[i + 1][2]
            features.update({
                'pos(+1)': next_pos,
                'chunk(+1)': next_chunk,
                'pos(0)+pos(+1)': pos + ' ' + next_pos,
                'chunk(0)+chunk(+1)': chunk + ' ' + next_chunk
            })
    else:
        features['EOS'] = True

    if i < len(sentence) - 2:
        next_word = sentence[i + 1][0]
        next_2_word = sentence[i + 2][0]

        features.update({
            'w(+2)': next_2_word,
            'w(+1)+w(+2)': next_word + ' ' + next_2_word,
            'isTitle(+2)': next_2_word.istitle(),
            'isNumber(+2)': next_2_word.isdigit()
        })

        if is_using_pos_chunk:
            next_pos = sentence[i + 1][1]
            next_2_pos = sentence[i + 2][1]
            next_chunk = sentence[i + 1][2]
            next_2_chunk = sentence[i + 2][2]
            features.update({
                'pos(+2)': next_2_pos,
                'chunk(+2)': next_2_chunk,
                'pos(+1)+pos(+2)': next_pos + ' ' + next_2_pos,
                'chunk(+1)+chunk(+2)': next_chunk + ' ' + next_2_chunk
            })

    return features


def get_features(sentence, is_using_pos_chunk):
    return [word_to_features(sentence, i, is_using_pos_chunk) for i in range(len(sentence))]


def get_sentences(paragraph):
    tokenized_sentences = [ViTokenizer.tokenize(sentence) for sentence in
                           re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s+', paragraph)]
    sentences = [[(token,) for token in re.sub(r'(?<=\d\s[/-])\s|(?=\s[/-]\s\d)\s', '', sentence).split()] for sentence
                 in tokenized_sentences]
    return sentences
