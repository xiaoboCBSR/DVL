import torch
from torch import nn
import torch.nn.functional as F


class Margins(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.input_dim = input_features    # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def forward(self, x, label, fc_type, scale, t, margin, decisive_margin):
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(x, weight_norm)  # x is l2-normalized
        cos_theta = cos_theta.clamp(-1, 1)    # for numerical stability
        batch_size = label.size(0)

        # for CRF testing, where label may equal to -1
        label = torch.where(label < 0, cos_theta.shape[1]-1, label)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # ground truth score

        if fc_type == 'NL':
            final_gt = torch.where(gt > 0, gt - margin, gt)
            cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        elif fc_type == 'DVL':
            # semi-hard vectors
            mask1 = cos_theta >= gt - decisive_margin
            mask2 = cos_theta <= gt
            mask = mask1 & mask2
            semi_hard_vector = cos_theta[mask]
            cos_theta[mask] = (t + 1.0) * semi_hard_vector + t
            final_gt = torch.where(gt > 0, gt - margin, gt)
            cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        elif fc_type == 'Test':
            final_gt = gt
            cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        else:
            print("unknown method!")

        cos_theta *= scale
        return cos_theta
