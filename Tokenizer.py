import io
import json


class Tokenizer:
    def __init__(self, filename_dataset, filename_vocabulary, split_token):
        self.__filename_dataset = filename_dataset
        self.__split_token = split_token
        with io.open('vocabulary/' + filename_vocabulary, 'r', encoding='UTF-8') as file:
            self.word_index = json.load(file)
        self.index_word = []
        for word in self.word_index:
            self.index_word.append(word)
