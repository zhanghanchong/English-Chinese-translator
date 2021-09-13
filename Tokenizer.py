import io
import json
import torch
import os


def get_dataset_filename(language):
    return f'dataset/{language}.txt'


def get_vocabulary_filename(language):
    return f'vocabulary/{language}.json'


def get_words(sentence, split_token):
    sentence = sentence.rstrip('\n').lower()
    if split_token == '':
        return list(sentence)
    return sentence.split(split_token)


class Tokenizer:
    def __build_vocabulary(self):
        word_count = {}
        with io.open(get_dataset_filename(self.__language), 'r', encoding='UTF-8') as file:
            while 1:
                sentence = file.readline()
                if len(sentence) == 0:
                    break
                words = get_words(sentence, self.__split_token)
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
        self.__file = None
        if not os.path.exists(get_vocabulary_filename(language)):
            self.__build_vocabulary()
        with io.open(get_vocabulary_filename(language), 'r', encoding='UTF-8') as file:
            self.word_index = json.load(file)
        self.index_word = []
        for word in self.word_index:
            self.index_word.append(word)

    def get_sequence(self, words, sequence_length):
        sequence = torch.zeros(sequence_length)
        sequence[0] = self.word_index['<SOS>']
        for i in range(len(words)):
            if words[i] in self.word_index:
                sequence[i + 1] = self.word_index[words[i]]
            else:
                sequence[i + 1] = self.word_index['<UNK>']
        sequence[len(words) + 1] = self.word_index['<EOS>']
        for i in range(len(words) + 2, sequence_length):
            sequence[i] = self.word_index['<PAD>']
        return sequence

    def get_batch(self, batch_size):
        if self.__file is None:
            self.__file = io.open(get_dataset_filename(self.__language), 'r', encoding='UTF-8')
        words_list = []
        sequence_length = 0
        for _ in range(batch_size):
            sentence = self.__file.readline()
            if len(sentence) == 0:
                break
            words = get_words(sentence, self.__split_token)
            words_list.append(words)
            sequence_length = max(sequence_length, len(words))
        batch_size = len(words_list)
        if batch_size == 0:
            self.__file.close()
            self.__file = None
            return None
        sequence_length += 2
        batch = torch.zeros((batch_size, sequence_length), dtype=torch.int32)
        for i in range(batch_size):
            batch[i] = self.get_sequence(words_list[i], sequence_length)
        return batch
