import io
import json
import os
import torch
import wx
from torch import nn
from torch import optim
from MaskBuilder import MaskBuilder
from Tokenizer import Tokenizer
from Tokenizer import get_dataset_filename
from Tokenizer import PAD, SOS, EOS
from TranslatorModel import TranslatorModel


def get_model_filename(source_language, target_language):
    return f"model/{source_language}-{target_language}.pth"


class Gui(wx.Frame):
    def __set_parameters(self):
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

    def __train(self, event):
        source_language = self.__text_ctrl_train_source_language.GetValue()
        target_language = self.__text_ctrl_train_target_language.GetValue()
        epochs = self.__text_ctrl_train_epochs.GetValue()
        if not os.path.exists(get_dataset_filename(source_language)):
            self.__text_ctrl_train_log.SetValue('No dataset for source language.')
            return
        if not os.path.exists(get_dataset_filename(target_language)):
            self.__text_ctrl_train_log.SetValue('No dataset for target language.')
            return
        try:
            epochs = int(epochs)
        except:
            self.__text_ctrl_train_log.SetValue('"Epochs" should be a positive integer.')
            return
        if epochs <= 0:
            self.__text_ctrl_train_log.SetValue('"Epochs" should be a positive integer.')
            return
        self.__text_ctrl_train_log.Clear()
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
                self.__text_ctrl_train_log.AppendText(f"epoch {epoch + 1} batch {batch_id} loss {loss.item():.4f}\n")
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(model, model_filename)

    def __predict(self, event):
        source_language = self.__text_ctrl_predict_source_language.GetValue()
        target_language = self.__text_ctrl_predict_target_language.GetValue()
        source_sentence = self.__text_ctrl_predict_source_sentence.GetValue()
        model_filename = get_model_filename(source_language, target_language)
        if not os.path.exists(model_filename):
            self.__text_ctrl_predict_target_sentence.SetValue('No model.')
            return
        if len(source_sentence) == 0:
            self.__text_ctrl_predict_target_sentence.SetValue('Sentence should not be empty.')
            return
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
        self.__text_ctrl_predict_target_sentence.SetValue(target_sentence)

    def __init__(self):
        super().__init__(None, title='Translator', size=(800, 600))
        self.Center()
        panel = wx.Panel(self)
        self.__text_ctrl_train_source_language = wx.TextCtrl(panel)
        self.__text_ctrl_train_target_language = wx.TextCtrl(panel)
        self.__text_ctrl_train_epochs = wx.TextCtrl(panel)
        self.__button_train = wx.Button(panel, label='Train')
        self.__button_train.Bind(wx.EVT_BUTTON, self.__train)
        self.__text_ctrl_train_log = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        self.__text_ctrl_predict_source_language = wx.TextCtrl(panel)
        self.__text_ctrl_predict_target_language = wx.TextCtrl(panel)
        self.__button_predict = wx.Button(panel, label='Translate')
        self.__button_predict.Bind(wx.EVT_BUTTON, self.__predict)
        self.__text_ctrl_predict_source_sentence = wx.TextCtrl(panel)
        self.__text_ctrl_predict_target_sentence = wx.TextCtrl(panel)
        box_train_parameters = wx.BoxSizer()
        box_train_parameters.Add(wx.StaticText(panel, label='Source language:'), proportion=2,
                                 flag=wx.EXPAND | wx.ALL, border=5)
        box_train_parameters.Add(self.__text_ctrl_train_source_language, proportion=4,
                                 flag=wx.EXPAND | wx.ALL, border=5)
        box_train_parameters.Add(wx.StaticText(panel, label='Target language:'), proportion=2,
                                 flag=wx.EXPAND | wx.ALL, border=5)
        box_train_parameters.Add(self.__text_ctrl_train_target_language, proportion=4,
                                 flag=wx.EXPAND | wx.ALL, border=5)
        box_train_parameters.Add(wx.StaticText(panel, label='Epochs:'), proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_train_parameters.Add(self.__text_ctrl_train_epochs, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_train_parameters.Add(self.__button_train, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_train_log = wx.BoxSizer()
        box_train_log.Add(self.__text_ctrl_train_log, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_parameters = wx.BoxSizer()
        box_predict_parameters.Add(wx.StaticText(panel, label='Source language:'), proportion=1,
                                   flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_parameters.Add(self.__text_ctrl_predict_source_language, proportion=3,
                                   flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_parameters.Add(wx.StaticText(panel, label='Target language:'), proportion=1,
                                   flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_parameters.Add(self.__text_ctrl_predict_target_language, proportion=3,
                                   flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_parameters.Add(self.__button_predict, proportion=1,
                                   flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_sentences = wx.BoxSizer()
        box_predict_sentences.Add(self.__text_ctrl_predict_source_sentence, proportion=1,
                                  flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_sentences.Add(self.__text_ctrl_predict_target_sentence, proportion=1,
                                  flag=wx.EXPAND | wx.ALL, border=5)
        box_main = wx.BoxSizer(wx.VERTICAL)
        box_main.Add(box_train_parameters, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_main.Add(box_train_log, proportion=5, flag=wx.EXPAND | wx.ALL, border=5)
        box_main.Add(box_predict_parameters, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_main.Add(box_predict_sentences, proportion=5, flag=wx.EXPAND | wx.ALL, border=5)
        panel.SetSizer(box_main)
        self.__set_parameters()


app = wx.App()
gui = Gui()
gui.Show()
app.MainLoop()
