import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertModel

from dataset import prepare_data_for_training
from model import SiameseModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
print(device)

bert_model = BertModel.from_pretrained('bert-base-uncased')
siam_model = SiameseModel(bert_model)

learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
num_epochs = 300
model_saving_path = './bert_siamese.pt'


def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_saving_path):
    since = time.time()
    tb = SummaryWriter()
    max_f1_score = -1
    for epoch in tqdm(range(num_epochs)):
        loss_dict = {}
        acc_dict = {}
        f1score_dict = {}
        for phase in ['train', 'valid']:
            preds_list = []
            labels_lit = []
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for _, data in enumerate(dataloaders_dict[phase], 0):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    sent_a = data['sent1_ids'].to(device, dtype=torch.long)
                    sent_b = data['sent2_ids'].to(device, dtype=torch.long)
                    labels = data['target'].to(device, dtype=torch.long)

                    outputs = model(sent_a, sent_b)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels)
                preds_list.append(preds.cpu().numpy())
                labels_lit.append(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1score = f1_score(np.concatenate(preds_list), np.concatenate(labels_lit), average='weighted')
            print('Loss', epoch_loss)
            print('F1 score', epoch_f1score)
            loss_dict[phase] = epoch_loss
            f1score_dict[phase] = epoch_f1score
            if epoch_f1score > max_f1_score:
                torch.save(model.state_dict(), model_saving_path)
                max_f1_score = epoch_f1score

        tb.add_scalars('Loss: epoch', {'Train': loss_dict['train'], 'Valid': loss_dict['valid']}, epoch)
        tb.add_scalars('F1 score: epoch', {'Train': f1score_dict['train'], 'Valid': f1score_dict['valid']}, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    train_loader, test_loader = prepare_data_for_training()
    torch.cuda.empty_cache()
    siam_model.to(device)
    optimizer = torch.optim.Adam(siam_model.parameters(), lr=learning_rate)
    dataloaders_dict = {'train': train_loader, 'valid': test_loader}
    model_ft = train_model(siam_model, dataloaders_dict, criterion, optimizer, num_epochs, model_saving_path)
