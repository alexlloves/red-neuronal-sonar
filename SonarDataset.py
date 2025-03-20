import torch
from torch.utils.data import Dataset
import pandas as pd
from StandardScaler import StandardScaler
from torch.utils.data import Dataset

class SonarDataset(Dataset):
    def __init__(self, src_file, root_dir, transform=None):
        fake_names = [f"Atribute{i}" for i in range(60)] + ["class"]

        sonarDataset = pd.read_csv(src_file, names=fake_names)

        X = sonarDataset.iloc[:, :60]
        Y = sonarDataset["class"]

        nomeClases = Y.unique()
        conversion = {v: k for k, v in enumerate(nomeClases)}
        YConversion = pd.DataFrame()
        for nome in nomeClases:
            YConversion[nome] = (Y == nome).astype(float)
        y_tensor = torch.as_tensor(YConversion.values).float()

        x_tensor = torch.tensor(X.values).float()
        scaler = StandardScaler()
        XScalada = torch.tensor(scaler.fit_transform(x_tensor)).float()

        self.data = torch.cat((XScalada, y_tensor), 1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        preds = self.data[idx, :60]  
        spcs = self.data[idx, 60:] 
        sample = (preds, spcs)
        if self.transform:
            sample = self.transform(sample)
        return sample
