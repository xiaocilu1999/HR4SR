from torch import nn


class Loss(object):
    def __init__(self, args, model):
        self.model = model
        self.theta = args.theta
        self.CE_Loss = nn.CrossEntropyLoss(ignore_index=0)

    def calculate_loss(self, sequence, label):
        denoise_output, feature_output = self.model(sequence)

        denoise_output = denoise_output.view(-1, denoise_output.size(-1))
        feature_output = feature_output.view(-1, feature_output.size(-1))
        label = label.view(-1)

        denoise_loss = self.CE_Loss(denoise_output, label)
        feature_loss = self.CE_Loss(feature_output, label)

        total_loss = self.theta * denoise_loss + (1 - self.theta) * feature_loss

        return total_loss
