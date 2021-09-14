import io
import json
import os
import torch
from torch import nn
from torch import optim
from MaskBuilder import MaskBuilder
from Tokenizer import Tokenizer
from Tokenizer import PAD, SOS, EOS
from TranslatorModel import TranslatorModel


def get_model_filename(source_language, target_language):
    return f"model/{source_language}-{target_language}.pth"


class Gui:
    def __init__(self):
        with io.open('parameters.json', 'r') as file:
            parameters = json.load(file)
        self.__adam_beta1 = parameters['adam_beta1']
        self.__adam_beta2 = parameters['adam_beta2']
        self.__adam_epsilon = parameters['adam_epsilon']
        self.__batch_size = parameters['batch_size']
        self.__d_model = parameters['d_model']
        self.__dim_feedforward = parameters['dim_feedforward']
        self.__dropout = parameters['dropout']
        self.__learning_rate = parameters['learning_rate']
        self.__nhead = parameters['nhead']
        self.__num_encoder_layers = parameters['num_encoder_layers']
        self.__num_decoder_layers = parameters['num_decoder_layers']
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, source_language, target_language, epochs):
        model_filename = get_model_filename(source_language, target_language)
        tokenizer = {source_language: Tokenizer(source_language), target_language: Tokenizer(target_language)}
        max_sequence_length = 0
        while 1:
            source = tokenizer[source_language].get_batch(self.__batch_size)
            target = tokenizer[target_language].get_batch(self.__batch_size)
            if source is None and target is None:
                break
            max_sequence_length = max(max_sequence_length, max(source.shape[0], target.shape[0]))
        if os.path.exists(model_filename):
            model = torch.load(model_filename)
        else:
            model = TranslatorModel(self.__d_model, self.__dim_feedforward, self.__dropout, max_sequence_length,
                                    self.__nhead, self.__num_encoder_layers, self.__num_decoder_layers,
                                    len(tokenizer[source_language].index_word),
                                    len(tokenizer[target_language].index_word))
        model = model.to(self.__device)
        model.train()
        loss_function = nn.CrossEntropyLoss(ignore_index=PAD)
        optimizer = optim.Adam(model.parameters(), self.__learning_rate, (self.__adam_beta1, self.__adam_beta2),
                               self.__adam_epsilon)
        for epoch in range(epochs):
            batch_id = 0
            while 1:
                source = tokenizer[source_language].get_batch(self.__batch_size)
                target = tokenizer[target_language].get_batch(self.__batch_size)
                if source is None and target is None:
                    break
                source = source.to(self.__device)
                target = target.to(self.__device)
                target_input = target[:-1]
                target_output = target[1:]
                mask_builder = MaskBuilder(source, target_input)
                source_mask = mask_builder.build_source_mask().to(self.__device)
                target_mask = mask_builder.build_target_mask().to(self.__device)
                source_padding_mask = mask_builder.build_source_padding_mask().to(self.__device)
                target_padding_mask = mask_builder.build_target_padding_mask().to(self.__device)
                logits = model(source, target_input, source_mask, target_mask, source_padding_mask, target_padding_mask,
                               source_padding_mask)
                optimizer.zero_grad()
                loss = loss_function(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
                loss.backward()
                optimizer.step()
                batch_id += 1
                print(f"epoch {epoch + 1} batch {batch_id} loss {loss.item():.4f}")
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(model, model_filename)

    def predict(self, source_language, target_language, source_sentence):
        model_filename = get_model_filename(source_language, target_language)
        if not os.path.exists(model_filename):
            return None
        tokenizer = {source_language: Tokenizer(source_language), target_language: Tokenizer(target_language)}
        model = torch.load(model_filename).to(self.__device)
        model.eval()
        source_words = tokenizer[source_language].get_words(source_sentence)
        source = tokenizer[source_language].get_sequence(source_words, len(source_words) + 2).to(self.__device)
        memory = model.encode(source)
        target = torch.zeros((1, 1), dtype=torch.int64, device=self.__device).fill_(SOS)
        while target[-1, 0] != EOS:
            target_mask = MaskBuilder(None, target).build_target_mask().to(self.__device)
            _, token = torch.max(model.decode(target, memory, target_mask)[-1, 0], dim=0)
            target = torch.cat(
                [target, torch.zeros((1, 1), dtype=torch.int64, device=self.__device).fill_(token.item())], dim=0)
        target_sentence = tokenizer[target_language].index_word[target[1, 0]]
        for i in range(2, target.shape[0] - 1):
            if target[i, 0] > EOS:
                target_sentence += ' ' + tokenizer[target_language].index_word[target[i, 0]]
        return target_sentence


gui = Gui()
print(gui.predict('Chinese', 'English', '我爱你。'))
