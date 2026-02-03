import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
scores_df = pd.read_csv('https://content-media-cdn.codefinity.com/courses/1dd2b0f6-6ec0-40e6-a570-ed0ac2209666/section_2/hours_scores.csv')
X = scores_df['Hours Studied'].values
Y = scores_df['Test Score'].values
# Convert to PyTorch tensors and reshape
X_tensor = torch.tensor(scores_df['number of hours'], dtype=torch.float32).reshape(N,1)
Y_tensor = torch.tensor(scores_df['test scores'], dtype=torch.float32).reshape(N,1)
# Define the linear regression model
model = nn.Linear(in_features=1, out_features= 1)
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), ln=0.01)
# Train the model
epochs = 100
for epoch in range(epochs):
    # Perform forward pass
    predictions = model(X_tensor)
    loss = criterion(predictions, Y_tensor)
    # Reset the gradient
    optimizer.zero_grad()
    # Perform backward pass
    loss.backward()
    # Update the parameters
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
      
# Print the model's parameters
weights = model.weight.data
bias = model.bias.data
print(f"Trained weights: {weights.item():.4f}")
print(f"Trained bias: {bias.item():.4f}")