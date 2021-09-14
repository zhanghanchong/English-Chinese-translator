import torch


class MaskBuilder:
    def __init__(self, source, target, padding_index):
        self.__source = source
        self.__target = target
        self.__padding_index = padding_index

    def build_source_mask(self):
        return torch.zeros((self.__source.shape[0], self.__source.shape[0]))

    def build_target_mask(self):
        target_mask = torch.tril(torch.ones((self.__target.shape[0], self.__target.shape[0])))
        return target_mask.masked_fill(target_mask == 0, float('-inf')).masked_fill(target_mask == 1, 0)

    def build_source_padding_mask(self):
        return (self.__source == self.__padding_index).transpose(0, 1)

    def build_target_padding_mask(self):
        return (self.__target == self.__padding_index).transpose(0, 1)
