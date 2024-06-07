import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import yaml
from yolov5 import train

class CustomDataset(Dataset):
    def __init__(self, root_dirs, class_labels, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = set()

        for i, dirs in enumerate(root_dirs):
            for dir_path in dirs:
                for img_name in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img_name)
                    self.samples.append((img_path, i))  
                    self.labels.add(class_labels[i])

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

root_dirs = [
    ['/Users/jiayu/Desktop/42/Blood/class0'],  # replace with actual paths
    ['/Users/jiayu/Desktop/42/Blood/class1']   # replace with actual paths
]
class_labels = ['benign', 'malignant']
dataset = CustomDataset(root_dirs, class_labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

data_config = {
    'train': '/Users/jiayu/Desktop/42/Blood/train',  # replace with actual paths
    'val': '/Users/jiayu/Desktop/42/Blood/val',      # replace with actual paths
    'nc': 2,
    'names': class_labels
}

with open('data.yaml', 'w') as file:
    yaml.dump(data_config, file)

train.run(
    data='data.yaml',
    imgsz=640,
    batch_size=16,
    epochs=50,
    weights='yolov5s.pt',
    project='BreastCancerDetection',
    name='YOLOv5',
    exist_ok=True
)

trained_model_path = 'BreastCancerDetection/YOLOv5/weights/best.pt'
torch.save(trained_model_path, 'centralized_yolo_breast_cancer_model.pt')

print("Model training completed and saved as centralized_yolo_breast_cancer_model.pt")
