import io
import json
import os


def get_dataset_filename(language):
    return f'dataset/{language}.txt'


def get_vocabulary_filename(language):
    return f'vocabulary/{language}.json'


class Tokenizer:
    def __build_vocabulary(self):
        word_count = {}
        with io.open(get_dataset_filename(self.__language), 'r', encoding='UTF-8') as file:
            while 1:
                sentence = file.readline()
                if len(sentence) == 0:
                    break
                sentence = sentence.rstrip('\n').lower()
                words = list(sentence) if self.__split_token == '' else sentence.split(self.__split_token)
                for word in words:
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
        word_count_sorted = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
        vocabulary = {'<UNK>': 0, '<PAD>': 1, '<SOS>': 2, '<EOS>': 3}
        for i in range(len(word_count_sorted)):
            vocabulary[word_count_sorted[i][0]] = i + 4
        if not os.path.exists('vocabulary'):
            os.mkdir('vocabulary')
        with io.open(get_vocabulary_filename(self.__language), 'w', encoding='UTF-8') as file:
            file.write(json.dumps(vocabulary, indent=4, ensure_ascii=False))

    def __init__(self, language, split_token):
        self.__language = language
        self.__split_token = split_token
        if not os.path.exists(get_vocabulary_filename(language)):
            self.__build_vocabulary()
        with io.open(get_vocabulary_filename(language), 'r', encoding='UTF-8') as file:
            self.word_index = json.load(file)
        self.index_word = []
        for word in self.word_index:
            self.index_word.append(word)
