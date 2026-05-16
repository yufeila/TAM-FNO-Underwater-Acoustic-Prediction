import torch

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be N, H, W, C or N, H, W
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:sample_idx[0].shape[0]] + self.eps
                mean = self.mean[:sample_idx[0].shape[0]]

        # x is in shape of batch*n or n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

def save_normalizers(path, x_norm, y_norm, meta):
    torch.save({
        "x_mean": x_norm.mean.cpu(),
        "x_std":  x_norm.std.cpu(),
        "y_mean": y_norm.mean.cpu(),
        "y_std":  y_norm.std.cpu(),
        "eps": x_norm.eps,
        "meta": meta
    }, str(path))

def load_normalizers(path, device=None):
    data = torch.load(str(path), map_location='cpu')
    
    # Create dummy normalizers to hold the data
    x_norm = UnitGaussianNormalizer(torch.zeros(1), eps=data['eps'])
    x_norm.mean = data['x_mean']
    x_norm.std = data['x_std']
    
    y_norm = UnitGaussianNormalizer(torch.zeros(1), eps=data['eps'])
    y_norm.mean = data['y_mean']
    y_norm.std = data['y_std']
    
    if device:
        x_norm.mean = x_norm.mean.to(device)
        x_norm.std = x_norm.std.to(device)
        y_norm.mean = y_norm.mean.to(device)
        y_norm.std = y_norm.std.to(device)
        
    return x_norm, y_norm
