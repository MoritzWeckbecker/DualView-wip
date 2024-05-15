from utils.explainers import FeatureKernelExplainer, Explainer
import torch
from utils.data import FeatureDataset
import time
import os.path

class InnerProductExplainer(FeatureKernelExplainer):
    name = "InnerProductExplainer"
    def __init__(self, model, dataset, device):
        super(InnerProductExplainer, self).__init__(model, dataset, device, normalize=False)
        self.coefficients = torch.ones(size=self.samples.shape)

# same as FeatureKernelExplainer, but with added normalisation in explain step to get cosine instead of simple inner product
# probably nicer to have this merged with FeatureKernelExplainer with a parameter to toggle the additional normalization on/off
class CosineFeatureKernelExplainer(Explainer):
    def __init__(self, model, dataset, device, file=None,normalize=True):
        super().__init__(model, dataset, device)
        # self.sanity_check = sanity_check
        if file is not None:
            if not os.path.isfile(file) and not os.path.isdir(file):
                file = None
        feature_ds = FeatureDataset(self.model, dataset, device, file)
        self.coefficients = None  # the coefficients for each training datapoint x class
        self.learned_weights = None
        self.normalize=normalize
        self.samples = feature_ds.samples.to(self.device)
        self.mean = self.samples.sum(0) / self.samples.shape[0]
        #self.mean = torch.zeros_like(self.mean)
        self.stdvar = torch.sqrt(torch.sum((self.samples - self.mean) ** 2, dim=0) / self.samples.shape[0])
        #self.stdvar=torch.ones_like(self.stdvar)
        self.normalized_samples=self.normalize_features(self.samples) if normalize else self.samples
        self.labels = torch.tensor(feature_ds.labels, dtype=torch.int, device=self.device)

    def normalize_features(self, features):
        return (features - self.mean) / self.stdvar

    def explain(self, x, preds=None, targets=None):
        assert self.coefficients is not None
        x = x.to(self.device)
        f = self.model.features(x)
        if self.normalize:
            f = self.normalize_features(f)
        crosscorr = torch.matmul(f, self.normalized_samples.T)
        #
        # added part
        #
        normalize_matrix = torch.outer(f.norm(p=2,dim=1), self.normalized_samples.norm(p=2,dim=1))
        crosscorr = torch.div(crosscorr, normalize_matrix)
        #
        # added part
        #
        crosscorr = crosscorr[:, :, None]
        xpl = self.coefficients * crosscorr
        indices = preds[:, None, None].expand(-1, self.samples.shape[0], 1)
        xpl = torch.gather(xpl, dim=-1, index=indices)
        return torch.squeeze(xpl)

    def save_coefs(self, dir):
        torch.save(self.coefficients, os.path.join(dir, f"{self.name}_coefs"))

# same as FeatureKernelExplainer, but with negative of l2 distance instead
class DistanceFeatureKernelExplainer(Explainer):
    name="DistanceFeatureKernelExplainer"
    @staticmethod
    def distance_norm(t1,t2):
        norm_t1 = torch.norm(t1,p=2,dim=1).square().unsqueeze(1).repeat(1, t2.shape[0])
        norm_t2 = torch.norm(t2,p=2,dim=1).square().unsqueeze(0).repeat(t1.shape[0], 1)
        t_out = norm_t1 + norm_t2 - 2 * t1 @ t2.T
        return t_out

    def __init__(self, model, dataset, device, file=None,normalize=True):
        super().__init__(model, dataset, device)
        # self.sanity_check = sanity_check
        if file is not None:
            if not os.path.isfile(file) and not os.path.isdir(file):
                file = None
        feature_ds = FeatureDataset(self.model, dataset, device, file)
        self.coefficients = None  # the coefficients for each training datapoint x class
        self.learned_weights = None
        self.normalize=normalize
        self.samples = feature_ds.samples.to(self.device)
        self.mean = self.samples.sum(0) / self.samples.shape[0]
        #self.mean = torch.zeros_like(self.mean)
        self.stdvar = torch.sqrt(torch.sum((self.samples - self.mean) ** 2, dim=0) / self.samples.shape[0])
        #self.stdvar=torch.ones_like(self.stdvar)
        self.normalized_samples=self.normalize_features(self.samples) if normalize else self.samples
        self.labels = torch.tensor(feature_ds.labels, dtype=torch.int, device=self.device)

    def normalize_features(self, features):
        return (features - self.mean) / self.stdvar

    def explain(self, x, preds=None, targets=None):
        x = x.to(self.device)
        f = self.model.features(x)
        if self.normalize:
            f = self.normalize_features(f)
        xpl = self.distance_norm(f, self.normalized_samples)
        return torch.squeeze(xpl)

    def save_coefs(self, dir):
        torch.save(self.coefficients, os.path.join(dir, f"{self.name}_coefs"))

class CosineInnerProductExplainer(CosineFeatureKernelExplainer):
    name = "InnerProductExplainerCosine"
    def __init__(self, model, dataset, device):
        super(CosineInnerProductExplainer, self).__init__(model, dataset, device, normalize=False)
        self.coefficients = torch.ones(size=self.samples.shape)