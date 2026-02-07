r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torch.utils.data.distributed import DistributedSampler as Sampler
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

from .deepglobe import DatasetDeepglobe_multiclass
from .isic import DatasetISIC_multiclass
# from .lung import DatasetLung
from .suim import DatasetSUIM_multiclass
import albumentations as A

class CDFSSDataset_multiclass:

    @classmethod
    def initialize(cls, img_size, datapath):

        cls.datasets = {
            'deepglobe': DatasetDeepglobe_multiclass,
            'isic': DatasetISIC_multiclass,
            # 'lung': DatasetLung,
            'suim': DatasetSUIM_multiclass,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)
                                           ]
                                          )
        cls.img_transform = A.Compose([A.Resize(img_size, img_size)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz=1, nworker=0, fold=0, split='val', shot=1, way=1, device=None):
        nworker = nworker

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold,
                                          transform=cls.transform,
                                          img_transform= cls.img_transform,
                                          split=split, shot=shot, way=way, device=device)
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        # train_sampler = Sampler(dataset) if split == 'trn' else None
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, num_workers=nworker,
                                pin_memory=True)

        return dataloader
