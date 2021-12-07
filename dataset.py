import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SICKDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.sent_1 = self.dataset.sentence_A
        self.sent_2 = self.dataset.sentence_B
        self.y = self.dataset.target
        self.source_len = 256
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.sent_1)

    def __getitem__(self, index):
        sent_a = str(self.sent_1[index])
        sent_a = ' '.join(sent_a.split())

        sent_b = str(self.sent_2[index])
        sent_b = ' '.join(sent_b.split())

        encod_1 = self.tokenizer.batch_encode_plus([sent_a], max_length=self.source_len,
                                                  pad_to_max_length=True, return_tensors='pt')
        encod_2 = self.tokenizer.batch_encode_plus([sent_b], max_length=self.source_len,
                                                  pad_to_max_length=True, return_tensors='pt')

        sent1_ids = encod_1['input_ids'].squeeze()
        sent2_ids = encod_2['input_ids'].squeeze()
        target = torch.tensor(self.y[index])

        return {
            'sent1_ids': sent1_ids.to(dtype=torch.long),
            'sent2_ids': sent2_ids.to(dtype=torch.long),
            'target': target
        }


def prepare_data_for_training():
    train_data = SICKDataset('data/train.csv')
    test_data = SICKDataset('data/test.csv')

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)
    return train_loader, test_loader