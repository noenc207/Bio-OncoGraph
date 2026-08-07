import torch
import torch.nn as nn
import numpy as np

class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, valid_loader, device='cuda', max_iter=50, lr=0.01):
        nll_criterion = nn.CrossEntropyLoss().to(device)
        
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        
        return self

class PlattScaling:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression()

    def fit(self, logits, labels):
        self.lr.fit(logits.reshape(-1, 1) if logits.ndim == 1 else logits, labels)

    def predict_proba(self, logits):
        return self.lr.predict_proba(logits.reshape(-1, 1) if logits.ndim == 1 else logits)

class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def reliability_diagram(logits, labels, n_bins=15):
    softmaxes = torch.nn.functional.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    diagram_data = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean().item()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            diagram_data.append({
                'bin': (bin_lower.item(), bin_upper.item()),
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'proportion': prop_in_bin
            })
    return diagram_data
