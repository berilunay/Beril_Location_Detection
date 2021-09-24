import torch
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from torchvision.datasets import VisionDataset
import cv2
from torch.utils.data import DataLoader

from colon_ai.traınıng.DataModelColon import ColonDataModel


class InferenceDatasetQuality(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, num_samples=None):
        super(InferenceDatasetQuality, self).__init__(root, transform=transform, target_transform=target_transform)
        self.video_dirs = []
        self.video_dirs.append(self.root)
        self.sample_dirs = []  # image0,image1 ...
        for video_dir in self.video_dirs:
            self.sample_dirs += (sorted(Path(video_dir).iterdir())) # gets image and labels(folder)
        print(self.sample_dirs)
        if num_samples:
            self.sample_dirs = self.sample_dirs[:num_samples]



    def __getitem__(self, index):
        sample_dir = self.sample_dirs[index]

        colon= self._load_and_transform_colon(sample_dir)

        return colon


    def _load_and_transform_colon(self, sample_dir):
        colon = self._load_colon(sample_dir)
        colon = self._apply_transforms(colon)

        return colon


    def _load_colon(self, sample_dir):

        im_path = str(sample_dir / "colon.png")
        colon=cv2.imread(im_path)
        resized_image = cv2.resize(colon, (224, 224),interpolation=cv2.INTER_NEAREST)
        img_new=cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        colon = np.array(img_new)
        colon=np.moveaxis(colon,-1,0)
        colon = torch.from_numpy(colon)
        colon= colon/255.0

        return colon


    def _apply_transforms(self, colon):
        if self.transform is not None:
            colon = self.transform(colon)

        return colon


    def __len__(self):
        return len(self.sample_dirs)

def show_ouput(model, dataloader, class_dict=None):

    for  features in dataloader:
        with torch.no_grad():
            features = features
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
        break

    fig, axes = plt.subplots(nrows=2, ncols=5,
                             sharex=True, sharey=True)


    nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))

    if nhwc_img.shape[-1] == 1:
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap='binary')
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]}')
            ax.axison = False

    else:

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhwc_img[idx])
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}')
            ax.axison = False
    plt.tight_layout()
    plt.show()

#def calculation(model,dataloader,dict):



if __name__ == '__main__':
    print("hello")
    #dataset = InferenceDatasetQuality('/home/beril/Thesis_Beril/Train_Labels_Quality/Video2')
    #colon= dataset[0]
    Test_Path="/home/beril/Thesis_Beril/Train_Labels_Quality/Video2"
    quality_dict = {0: 'G', 1: 'M', 2: 'p', 3: 'B'}
    test_dataset = InferenceDatasetQuality(root=Test_Path)
    test_loader = DataLoader(test_dataset,batch_size=10)
    checkpoint_model_path="/home/beril/BerilCodes/ColonAI_LocationDetection/colon_ai/traınıng/uncategorized/2jsc8uzm/checkpoints/epoch=64-val_loss=0.53-val_acc=0.83.ckpt"
    pretrained_model = ColonDataModel.load_from_checkpoint(checkpoint_path= checkpoint_model_path)
    pretrained_model.eval()
    pretrained_model.freeze()
    show_ouput(pretrained_model,test_loader,quality_dict)




