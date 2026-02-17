import torch
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_data(name, batch_size):
    if name == "CIFAR100":
        t = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train = datasets.CIFAR100('./data', train=True, download=True, transform=t)
        test = datasets.CIFAR100('./data', train=False, download=True, transform=t)
        return DataLoader(train, batch_size, shuffle=True), DataLoader(test, batch_size), 3072, 3, 100, 32

    elif name == "Adult":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        cols = ['age','workclass','fnlwgt','education','edu_num','marital','occupation','relationship','race','sex','cap_gain','cap_loss','hpw','country','income']
        df = pd.read_csv(url, names=cols, skipinitialspace=True).dropna()
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])
        X = StandardScaler().fit_transform(df.drop('income', axis=1).values)
        y = le.fit_transform(df['income']).astype(np.int64)
        
        # PADDING: 14 -> 16 for CNN 4x4
        X = np.pad(X, ((0,0), (0,2)), mode='constant')
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        train_size = int(0.8 * len(dataset))
        train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
        return DataLoader(train_ds, batch_size, shuffle=True), DataLoader(test_ds, batch_size), 16, 1, 2, 4

    elif name == "PCam":
        t = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.701, 0.538, 0.691), (0.235, 0.277, 0.213))
        ])
        
        print("Loading PCam... This may take a while if downloading for the first time.")
        full_train = datasets.PCAM('./data', split='train', download=True, transform=t)
        full_test = datasets.PCAM('./data', split='test', download=True, transform=t)
        
        # Taking a Subset to stay within the 1-hour per model limit
        train_indices = np.random.choice(len(full_train), 20000, replace=False)
        test_indices = np.random.choice(len(full_test), 5000, replace=False)
        
        train_ds = Subset(full_train, train_indices)
        test_ds = Subset(full_test, test_indices)
        
        return DataLoader(train_ds, batch_size, shuffle=True), DataLoader(test_ds, batch_size), 3072, 3, 2, 32
