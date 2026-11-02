"""
Triage AI - Deep Learning Module (GitHub Issue #4)
PyTorch Neural Network monitoring continuous ICU patient vitals 
to predict acute physiological deterioration.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)

print("--- 1. Generating Continuous ICU Telemetry Data ---")
# Features: [Heart Rate (BPM), Mean Arterial Pressure (mmHg), Resp Rate (bpm), SpO2 (%)]
# 2,000 synthetic patient telemetry observation windows
n_samples = 2000
X = np.empty((n_samples, 4), dtype=np.float32)

# Normal vitals baseline
X[:, 0] = np.random.normal(75, 10, n_samples)   # Heart Rate
X[:, 1] = np.random.normal(85, 8, n_samples)    # MAP
X[:, 2] = np.random.normal(16, 3, n_samples)    # Resp Rate
X[:, 3] = np.random.normal(98, 1.5, n_samples)  # SpO2

# Label: 1 = Deterioration / Sepsis risk, 0 = Stable
# Deterioration when HR > 100, RR > 22, SpO2 < 92
y = ((X[:, 0] > 95) & (X[:, 2] > 20) | (X[:, 3] < 92)).astype(np.float32).reshape(-1, 1)

# Split 80/20
train_size = int(0.8 * n_samples)
X_train, X_test = torch.tensor(X[:train_size]), torch.tensor(X[train_size:])
y_train, y_test = torch.tensor(y[:train_size]), torch.tensor(y[train_size:])

print(f"Data Prepared: {train_size} training samples, {n_samples - train_size} test samples.")

print("\n--- 2. Defining Deep Neural Network Architecture ---")
class VitalsMonitorNet(nn.Module):
    def __init__(self):
        super(VitalsMonitorNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

model = VitalsMonitorNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\n--- 3. Training Deep Learning Model (50 Epochs) ---")
for epoch in range(1, 51):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/50 - Loss: {loss.item():.4f}")

# Evaluation on test set
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_classes = (test_preds > 0.5).float()
    accuracy = (test_classes.eq(y_test).sum().item()) / y_test.size(0)

print(f"\nTest Deterioration Classification Accuracy: {accuracy * 100:.2f}%")
print("[SUCCESS] Deep Learning vitals monitoring module completed.")