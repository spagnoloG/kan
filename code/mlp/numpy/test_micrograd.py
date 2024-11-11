from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from micrograd.engine import Value
from micrograd.nn import MLP
import numpy as np

data = fetch_california_housing()
X = data["data"]
y = data["target"].reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

mlp = MLP(nin=8, nouts=[20, 10, 1])  # Hidden layers are similar to NN in your code

# Training loop
epochs = 5
learning_rate = 0.01

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X_train)):
        # Forward pass
        inputs = [Value(x) for x in X_train[i]]
        target = Value(y_train[i][0])

        # Predictions
        output = mlp(inputs)
        loss = (output - target) ** 2  # Mean Squared Error loss

        # Backward pass and parameter update
        mlp.zero_grad()
        loss.backward()

        for p in mlp.parameters():
            p.data -= learning_rate * p.grad  # Gradient descent step

        total_loss += loss.data

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(X_train)}")

# Testing the model
predictions = []
for x in X_test:
    inputs = [Value(val) for val in x]
    output = mlp(inputs)
    predictions.append(output.data)

# Inverse transform predictions and calculate MSE
predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_original, predictions)
print(f"Mean Squared Error on California Housing dataset: {mse:.2f}")
