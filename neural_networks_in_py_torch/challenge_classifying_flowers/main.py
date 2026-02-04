import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load data and split
torch.manual_seed(42)
iris_df = pd.read_csv(
    "https://content-media-cdn.codefinity.com/"
    "courses/1dd2b0f6-6ec0-40e6-a570-ed0ac2209666/section_2/iris.csv"
)
X = iris_df.drop(columns=["species"]).values
y = iris_df["species"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Create tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

# 3. Define the model
class IrisModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)   # raw logits for CrossEntropy
        return x

input_size = X.shape[1]
hidden_size = 16
output_size = len(iris_df["species"].unique())

model = IrisModel(input_size, hidden_size, output_size)

# 4. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 6. Evaluation
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_test_pred_labels = torch.argmax(y_test_pred, dim=1)

accuracy = (y_test_pred_labels == y_test_tensor).float().mean().item() * 100
print(f"Test accuracy: {accuracy:.2f}%")