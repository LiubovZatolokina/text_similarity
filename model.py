import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class SiameseModel(nn.Module):

    def __init__(self, _model):
        super(SiameseModel, self).__init__()
        self.hidden_size = 256
        self.bert = _model
        self.bert.to(device)
        self.dropout = nn.Dropout(0.3)
        self.num_labels = 3
        self.linear_1 = nn.Linear(768 * 3, self.hidden_size * 3)
        self.linear_2 = nn.Linear(self.hidden_size * 3, self.num_labels)

    def forward(self, a_input_ids, b_input_ids):
        a_outputs = self.bert(input_ids=a_input_ids)
        b_outputs = self.bert(input_ids=b_input_ids)
        concated_pooled_output = torch.cat(
            [a_outputs[1], b_outputs[1], torch.abs(a_outputs[1] - b_outputs[1])], dim=1)
        concated_pooled_output = self.dropout(concated_pooled_output)
        output_linear = self.linear_1(concated_pooled_output)
        logits = self.linear_2(output_linear)
        return logits
