import io
import json


class VocabularyBuilder:
    def __init__(self, filename_dataset, filename_vocabulary, split_token):
        self.__filename_dataset = filename_dataset
        self.__filename_vocabulary = filename_vocabulary
        self.__split_token = split_token

    def build(self):
        word_count = {}
        with io.open('dataset/' + self.__filename_dataset, 'r', encoding='UTF-8') as file:
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
        with io.open('vocabulary/' + self.__filename_vocabulary, 'w', encoding='UTF-8') as file:
            file.write(json.dumps(vocabulary, indent=4, ensure_ascii=False))
