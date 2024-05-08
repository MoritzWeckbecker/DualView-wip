from utils.explainers import FeatureKernelExplainer
import torch

class InnerProductExplainer(FeatureKernelExplainer):
    name = "InnerProductExplainer"
    def __init__(self, model, dataset, device):
        super(InnerProductExplainer, self).__init__(model, dataset, device, normalize=False)
        self.coefficients = torch.ones(size=self.samples.shape)