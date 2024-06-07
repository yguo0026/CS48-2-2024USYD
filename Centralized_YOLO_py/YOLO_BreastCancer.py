
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from yolov5 import train

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        # Iterate over patient ID folders
        for patient_id in sorted(os.listdir(root)):
            patient_path = os.path.join(root, patient_id)
            for class_label in ['0', '1']:
                class_path = os.path.join(patient_path, class_label)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        label = int(class_label)
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

dataset = CustomDataset(root='/Users/jiayu/Desktop/42/Breast', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

data_config = {
    'train': '/Users/jiayu/Desktop/42/Breast/train',
    'val': '/Users/jiayu/Desktop/42/Breast/val',
    'nc': 2,
    'names': ['benign', 'malignant']
}

with open('data.yaml', 'w') as file:
    yaml.dump(data_config, file)

train.run(
    data='data.yaml',  # Path to data config file
    imgsz=640,
    batch_size=16,
    epochs=50,
    weights='yolov5s.pt',  # Pre-trained weights, use 'yolov5s.pt', 'yolov5m.pt', etc.
    project='BreastCancerDetection',
    name='YOLOv5',
    exist_ok=True
)


trained_model_path = 'BreastCancerDetection/YOLOv5/weights/best.pt'
torch.save(trained_model_path, 'centralized_yolo_breast_cancer_model.pt')

print("Model training completed and saved as centralized_yolo_breast_cancer_model.pt")
