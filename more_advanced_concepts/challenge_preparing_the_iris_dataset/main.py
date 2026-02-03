import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

torch.manual_seed(42)
iris_df = pd.read_csv('https://content-media-cdn.codefinity.com/courses/1dd2b0f6-6ec0-40e6-a570-ed0ac2209666/section_2/iris.csv')
features = iris_df.drop(columns='species').values
target = iris_df['species'].values
# Convert features and target into PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.long)
# Create a TensorDataset
iris_dataset = TensorDataset(
    torch.tensor(features, dtype = torch.float32),
    torch.tensor(target,   dtype = torch.long)
)
# Wrap the dataset in a DataLoader
iris_loader = DataLoader(
    iris_dataset,
    batch_size= 32,
    shuffle= True
)
# Display the DataLoader in action
for batch_idx, (batch_features, batch_targets) in enumerate(iris_loader):
    print(f"Batch {batch_idx + 1}:")
    print("Features:\n", batch_features)
    print("Targets:\n", batch_targets)
    break