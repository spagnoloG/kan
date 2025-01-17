# Gasper Spagnolo, 2024

import numpy as np


class NN:
    def __init__(self, layers, learning_rate=0.1, batch_size=32):
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.weights = [
            np.random.normal(0.0, pow(layers[i], -0.5), (layers[i + 1], layers[i]))
            for i in range(len(layers) - 1)
        ]
        self.biases = [np.zeros((layers[i + 1], 1)) for i in range(len(layers) - 1)]

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_prime(self, a):
        return a * (1 - a)

    def forward(self, inputs):
        inputs = np.array(inputs, ndmin=2)
        activations = [inputs]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w.T) + b.T
            if i == len(self.weights) - 1:
                # Output layer uses linear activation
                a = z
            else:
                a = self.sigmoid(z)
            activations.append(a)
        return activations

    def cost_function(self, y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def cost_function_prime(self, y, t):
        return y - t

    def backprop(self, X, y, activations):
        batch_size = X.shape[0]
        errors = [self.cost_function_prime(activations[-1], y)]
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                # Output layer derivative is 1
                delta = errors[-1]
            else:
                delta = errors[-1] * self.sigmoid_prime(activations[i + 1])
            grad_w = np.dot(delta.T, activations[i]) / batch_size
            grad_b = np.mean(delta, axis=0, keepdims=True).T
            errors.append(np.dot(delta, self.weights[i]))
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b
        errors.reverse()

    def fit(self, inputs_list, targets_list):
        fit_loss = 0
        for i in range(0, len(inputs_list), self.batch_size):
            inputs_batch = np.array(inputs_list[i : i + self.batch_size], ndmin=2)
            targets_batch = np.array(targets_list[i : i + self.batch_size], ndmin=2)

            activations = self.forward(inputs_batch)
            self.backprop(inputs_batch, targets_batch, activations)
            fit_loss += self.cost_function(activations[-1], targets_batch)

        fit_loss /= len(inputs_list)

    def predict(self, inputs_list):
        return self.forward(inputs_list)[-1]

if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    import numpy as np

    # Load and preprocess data
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
    y_train = scaler_y.fit_transform(y_train).ravel()  # Flatten for scikit-learn compatibility
    y_test = scaler_y.transform(y_test).ravel()

    # Train NN 
    nn = NN(layers=[8, 20, 10, 1], learning_rate=0.1)
    epochs = 100
    for epoch in range(epochs):
        nn.fit(X_train, y_train.reshape(-1, 1))  # Reshape to keep consistency

    predictions_nn = []
    for x in X_test:
        output = nn.predict(x)
        predictions_nn.append(output[0, 0])

    predictions_nn = scaler_y.inverse_transform(np.array(predictions_nn).reshape(-1, 1))
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    mse_nn = mean_squared_error(y_test_original, predictions_nn)
    print(f"Neural Network Mean Squared Error: {mse_nn:.2f}")

    # Train scikit-learn Neural Network (MLPRegressor)
    mlp_reg = MLPRegressor(hidden_layer_sizes=(20, 10), learning_rate_init=0.1, max_iter=100, random_state=42)
    mlp_reg.fit(X_train, y_train)

    predictions_mlp = mlp_reg.predict(X_test).reshape(-1, 1)
    predictions_mlp = scaler_y.inverse_transform(predictions_mlp)

    mse_mlp = mean_squared_error(y_test_original, predictions_mlp)
    print(f"scikit-learn Neural Network Mean Squared Error: {mse_mlp:.2f}")

    # Train Linear Regression model
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)

    predictions_lr = linear_reg.predict(X_test)
    predictions_lr = scaler_y.inverse_transform(predictions_lr.reshape(-1, 1))

    mse_lr = mean_squared_error(y_test_original, predictions_lr)
    print(f"Linear Regression Mean Squared Error: {mse_lr:.2f}")
