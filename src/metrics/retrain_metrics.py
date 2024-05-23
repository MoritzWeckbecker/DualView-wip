from utils import Metric
import torch
import copy
from utils.data import RestrictedDataset
from train import start_training

class RetrainMetric(Metric):
    name = "RetrainMetric"
    
    def __init__(self, train, test, model, device="cuda"):
        self.train = train
        self.test = test



    def retrain(self, indices):
        #
        start_training(
            model_name=train_config.get('model_name', None),
            model_path=train_config.get('model_path', None),
            base_epoch=train_config.get('base_epoch', 0),
            device=self.device,
            num_classes=train_config.get('num_classes', None),
            class_groups=train_config.get('class_groups', None),
            dataset_name=train_config.get('dataset_name', None),
            dataset_type=train_config.get('dataset_type', 'std'),
            data_root=train_config.get('data_root', None),
            epochs=train_config.get('epochs', None),
            batch_size=train_config.get('batch_size', None),
            lr=train_config.get('lr', 0.1),
            augmentation=train_config.get('augmentation', None),
            loss=train_config.get('loss', None),
            optimizer=train_config.get('optimizer', None),
            save_dir=train_config.get('save_dir', None),
            save_each=train_config.get('save_each', 100),
            num_batches_eval=train_config.get('num_batches_eval', None),
            validation_size=train_config.get('validation_size', 2000),
            scheduler=train_config.get('scheduler', None),
            train_indices=indices
            )
        return retrained_model
    
# Parent Retrain Metric should only have method 'retrain' as scores etc are all different depending on the metric
    
class LeaveBatchOut(RetrainMetric):
    name = "LeaveBatchOutMetric"

    def __init__(self, train, test, model, batch_nr=10, device="cuda"):
        self.train = train
        self.test = test
        self.model = model
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.batch_nr = batch_nr
        self.batchsize = len(self.train) // batch_nr
        self.device = device

    def __call__(self, xpl):
        xpl.to(self.device)
        combined_xpl = xpl.sum(dim=0)
        indices_sorted = combined_xpl.argsort(descending=True)
        for i in range(self.batch_nr):
            # maybe write a load_restricted_dataset with indices instead and call it in start_training?
            ds = RestrictedDataset(self.train, indices_sorted[:(i+1)*self.batchsize], return_indices=True)
            evalds = self.test
            # retrain
            # correct accuracy on evalds

'''
        # copy model with blank weights
        retrained_model = copy.deepcopy(model)
        for layer in retrained_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # define retraining regiment
        retraining_loader = torch.utils.data.DataLoader(train_new, batch_size=4, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # retrain
        retrained_model.train()
        '''