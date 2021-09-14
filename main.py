import io
import json
import torch
from torch import nn
from torch import optim
from MaskBuilder import MaskBuilder
from Tokenizer import Tokenizer
from TranslatorModel import TranslatorModel


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

    def train(self, source_language, target_language, source_split_token, target_split_token, epochs):
        tokenizer = {source_language: Tokenizer(source_language, source_split_token),
                     target_language: Tokenizer(target_language, target_split_token)}
        padding_index = tokenizer[source_language].word_index['<PAD>']
        max_sequence_length = 0
        while 1:
            source = tokenizer[source_language].get_batch(self.__batch_size)
            target = tokenizer[target_language].get_batch(self.__batch_size)
            if source is None and target is None:
                break
            max_sequence_length = max(max_sequence_length, max(source.shape[0], target.shape[0]))
        model = TranslatorModel(self.__d_model, self.__dim_feedforward, self.__dropout, max_sequence_length,
                                self.__nhead, self.__num_encoder_layers, self.__num_decoder_layers,
                                len(tokenizer[source_language].index_word), len(tokenizer[target_language].index_word))
        model = model.to(self.__device)
        model.train()
        loss_function = nn.CrossEntropyLoss(ignore_index=padding_index)
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
                mask_builder = MaskBuilder(source, target_input, padding_index)
                source_mask, target_mask, source_padding_mask, target_padding_mask = mask_builder.build_masks()
                source_mask = source_mask.to(self.__device)
                target_mask = target_mask.to(self.__device)
                source_padding_mask = source_padding_mask.to(self.__device)
                target_padding_mask = target_padding_mask.to(self.__device)
                logits = model(source, target_input, source_mask, target_mask, source_padding_mask, target_padding_mask,
                               source_padding_mask)
                optimizer.zero_grad()
                loss = loss_function(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
                loss.backward()
                optimizer.step()
                batch_id += 1
                print(f"epoch {epoch + 1} batch {batch_id} loss {loss.item():.4f}")


gui = Gui()
gui.train('Chinese', 'English', '', ' ', 1)
