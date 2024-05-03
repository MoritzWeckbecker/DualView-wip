from torchvision import transforms
from torchvision.datasets import VisionDataset
from PIL import Image
import torch
import numpy as np
import os

class_names = dict()
class_names[0] = 'antelope' 
class_names[1] = 'bat'
class_names[2] = 'beaver'
class_names[3] = 'blue whale' 
class_names[4] = 'bobcat'
class_names[5] = 'buffalo' 
class_names[6] = 'chihuahua' 
class_names[7] = 'chimpanzee'
class_names[8] = 'collie' 
class_names[9] = 'cow'
class_names[10] = 'dalmation' 
class_names[11] = 'deer'
class_names[12] = 'dolphin'
class_names[13] = 'elephant' 
class_names[14] = 'fox'
class_names[15] = 'german shepherd' 
class_names[16] = 'giant panda' 
class_names[17] = 'giraffe'
class_names[18] = 'gorilla' 
class_names[19] = 'grizzly bear'
class_names[20] = 'hamster' 
class_names[21] = 'hippopotamus'
class_names[22] = 'horse'
class_names[23] = 'humpback whale' 
class_names[24] = 'killer whale'
class_names[25] = 'leopard' 
class_names[26] = 'lion' 
class_names[27] = 'mole'
class_names[28] = 'moose' 
class_names[29] = 'mouse'
class_names[30] = 'otter' 
class_names[31] = 'ox'
class_names[32] = 'persian cat'
class_names[33] = 'pig' 
class_names[34] = 'polar bear'
class_names[35] = 'rabbit' 
class_names[36] = 'raccoon' 
class_names[37] = 'rat'
class_names[38] = 'rhinoceros' 
class_names[39] = 'seal'
class_names[40] = 'sheep' 
class_names[41] = 'siamese cat'
class_names[42] = 'skunk'
class_names[43] = 'spider monkey' 
class_names[44] = 'squirrel'
class_names[45] = 'tiger' 
class_names[46] = 'walrus' 
class_names[47] = 'weasel'
class_names[48] = 'wolf' 
class_names[49] = 'zebra'

class_labels = list(class_names.values())

class AWA(VisionDataset):
    default_class_groups = [[i] for i in range(50)]
    name = 'AWA'
    default_transform = transforms.Compose([transforms.ToTensor()])
    inverse_transform = transforms.Compose([])
    
    def __init__(
        self,
        root="./AWA_rawdata",
        split="train",
        transform=None,
        inv_transform=None,
        target_transform=None, 
        #download=False,
        validation_size=2000
    ):
        if transform is None:
            transform=AWA.default_transform
        if inv_transform is None:
            inv_transform=AWA.inverse_transform
        train=(split=="train")
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train=train
        self.split=split
        self.inverse_transform=inv_transform #MUST HAVE THIS FOR MARK DATASET TO WORK
        self.classes=[i for i in range(50)]

        #val_input = np.squeeze(np.load(os.path.join(root, 'AWA_train_input.npy')))
        #val_label = np.squeeze(np.load(os.path.join(root, 'AWA_train_label.npy')))
        #test_input = np.squeeze(np.load(os.path.join(root, 'AWA_val_input.npy')))
        #test_label = np.squeeze(np.load(os.path.join(root, 'AWA_val_label.npy')))
        #self.data = np.concatenate((val_input, test_input), axis=0) #TOO LARGE :( MAKE THIS SMALLER?
        #self.targets = np.concatenate((val_label, test_label), axis=0)
        self.data = np.empty(shape=(37322,3,224,224), dtype=np.float32) #CHANGE SIZE WHEN USING LESS DATA, STILL TO LARGE
        self.targets = np.empty(shape=(37322))
        self.data[:29870,:,:,:] = np.squeeze(np.load(os.path.join(root, 'AWA_train_input.npy')))
        self.targets[:29870] = np.squeeze(np.load(os.path.join(root, 'AWA_train_label.npy')))
        self.data[29870:,:,:,:] = np.squeeze(np.load(os.path.join(root, 'AWA_val_input.npy')))
        self.data[29870:] = np.squeeze(np.load(os.path.join(root, 'AWA_val_label.npy')))
        
        N = len(self.targets)

        if not train:
            if (os.path.isfile("AWA_val_ids") and os.path.isfile("AWA_test_ids")):
                self.val_ids=torch.load("AWA_val_ids")
                self.test_ids=torch.load("AWA_test_ids")
            else:
                torch.manual_seed(42)  # THIS SHOULD NOT BE CHANGED BETWEEN TRAIN TIME AND TEST TIME
                perm = torch.randperm(N)
                self.val_ids = torch.tensor([i for i in perm[:validation_size]])
                self.test_ids = torch.tensor([i for i in perm[validation_size:]])
                torch.save(self.val_ids, 'AWA_val_ids')
                torch.save(self.test_ids, 'AWA_test_ids')

            print("Validation ids:")
            print(self.val_ids)
            print("Test ids:")
            print(self.test_ids)
            self.test_targets=torch.tensor(self.targets)[self.test_ids]

    def __getitem__(self, item):
        if self.split=="train":
            id=item
        elif self.split=="val":
            id=self.val_ids[item]
        else:
            id=self.test_ids[item]
        img, target = self.data[id], self.targets[id]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self):
        if self.split=="train":
            return len(self.targets)
        elif self.split=="val":
            return len(self.val_ids)
        else:
            return len(self.test_ids)