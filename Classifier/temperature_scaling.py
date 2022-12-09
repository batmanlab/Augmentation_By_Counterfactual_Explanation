import torch
from torch import nn, optim
from torch.nn import functional as F
import pdb

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, alpha = 0.0):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.alpha = alpha

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def get_focal_loss(self, logits, targets):
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        task_loss = criterion(logits.float(), targets.float())
        pt = torch.exp(-task_loss) # prevents nans when probability 0
        focal_loss = 0.25 * (1-pt)**self.alpha * task_loss
        task_loss =  focal_loss.mean()
        return task_loss

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.BCEWithLogitsLoss().cuda()#nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for samples in valid_loader:
                images = samples["img"].float().cuda()
                targets = samples["lab"].cuda()
                logits = self.model(images)
                logits_list.append(logits)
                try:
                    labels_list.append(targets)
                except:
                    pdb.set_trace()
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            mask = torch.isnan(labels)
            logits[mask] = 0
            labels[mask] = 0

        # Calculate NLL and ECE before temperature scaling
        if self.alpha == 0:
            before_temperature_nll = nll_criterion(logits, labels).item()
        else:
            before_temperature_nll = self.get_focal_loss(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.0001, max_iter=1000)

        def eval():
            if self.alpha == 0:
                loss = nll_criterion(self.temperature_scale(logits), labels)
            else:
                loss = self.get_focal_loss(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        if self.alpha == 0:
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        else:
            after_temperature_nll = self.get_focal_loss(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_cls = logits.shape[1]
        sigmoids = F.sigmoid(logits)
        confidences, predictions = torch.max(sigmoids, 1)
        _, true_labels = torch.max(labels, 1)
        accuracies = predictions.eq(true_labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece