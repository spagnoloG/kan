# MLP Backpropagation

## Forward Pass

Let’s assume a neural network with two layers. For simplicity, let:
    1. $\mathbf{x}$ \in $\mathbb{R}^n$ be the input vector.
    2. $\mathbf{W}_1$ \in $\mathbb{R}^{m \times n}$ and $\mathbf{b}_1$ \in $\mathbb{R}^m$ be the weight matrix and bias vector for the first layer.
    3. $\mathbf{W}_2$ \in $\mathbb{R}^{k \times m}$ and $\mathbf{b}_2 \in \mathbb{R}^k$ be the weight matrix and bias vector for the second (output) layer.
    4. $\sigma(\cdot)$ is an activation function (such as sigmoid or ReLU).

### Step 1: First Layer Computations
1. Compute the weighted sum of inputs for the first layer:
    $$
   \mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1
   $$
2. Apply the activation function to get the activations of the first layer:
    $$
   \mathbf{a}_1 = \sigma(\mathbf{z}_1)
    $$

###  2: Second Layer Computations (Output Layer)
1. Compute the weighted sum for the second layer:
    $$
   \mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2
   $$

2. Apply the activation function (or softmax if it’s for classification) to get the final output:
   $$
   \mathbf{y} = \sigma(\mathbf{z}_2)
   $$

Here, $\mathbf{y}$ is the network’s prediction for a given input $\mathbf{x}$.

---

## Backpropagation Using Chain Rule

Let’s assume a loss function $L(\mathbf{y}, \mathbf{t})$, where $\mathbf{t}$ is the target output vector. We’ll backpropagate to update $\mathbf{W}_2$, $\mathbf{b}_2$, $\mathbf{W}_1$, and $\mathbf{b}_1$ using gradients obtained via the chain rule.

### Step 1: Compute the Derivative of the Loss with respect to $\mathbf{z}_2$

1. The loss gradient with respect to the output $\mathbf{y}$ is:
    $$
   \frac{\partial L}{\partial \mathbf{y}} = \nabla_{\mathbf{y}} L
   $$
2. Using the chain rule, compute the derivative of L with respect to $\mathbf{z}_2$:
    $$
   \frac{\partial L}{\partial \mathbf{z}_2} = \frac{\partial L}{\partial \mathbf{y}} \circ \sigma'(\mathbf{z}_2)
   $$
   where $\circ$ denotes element-wise multiplication and $\sigma'(\mathbf{z}_2)$ is the derivative of the activation function applied element-wise to $\mathbf{z}_2$.

### Step 2: Derivatives with respect to $\mathbf{W}_2$ and $\mathbf{b}_2$

1. Using the chain rule, compute the gradient of the loss with respect to $\mathbf{W}_2$:
    $$
   \frac{\partial L}{\partial \mathbf{W}_2} = \frac{\partial L}{\partial \mathbf{z}_2} \cdot \mathbf{a}_1^\top
   $$
2. Similarly, compute the gradient with respect to $\mathbf{b}_2$:
    $$  
    \frac{\partial L}{\partial \mathbf{b}_2} = \frac{\partial L}{\partial \mathbf{z}_2}
    $$

#### Step 3: Backpropagate to the First Layer

Now, propagate the error back to the first layer to compute the gradients with respect to $\mathbf{W}_1$ and $\mathbf{b}_1$.

1. Compute the error at the first layer by backpropagating $\frac{\partial L}{\partial \mathbf{z}_2}$ through $\mathbf{W}_2$:
    $$  
   \frac{\partial L}{\partial \mathbf{z}_1} = (\mathbf{W}_2^\top \cdot \frac{\partial L}{\partial \mathbf{z}_2}) \circ \sigma'(\mathbf{z}_1)
   $$
2. Compute the gradient of the loss with respect to $\mathbf{W}_1$:
    $$
   \frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \cdot \mathbf{x}^\top
   $$
3. Compute the gradient with respect to $\mathbf{b}_1$:
    $$
   \frac{\partial L}{\partial \mathbf{b}_1} = \frac{\partial L}{\partial \mathbf{z}_1}
   $$

---

### Final Backpropagation Updates

With the gradients computed, update each parameter using a learning rate \eta:

1. Update weights and biases for the second layer:
    $$
   \mathbf{W}_2 := \mathbf{W}_2 - \eta \frac{\partial L}{\partial \mathbf{W}_2}
   $$
   $$
   \mathbf{b}_2 := \mathbf{b}_2 - \eta \frac{\partial L}{\partial \mathbf{b}_2}
    $$

2. Update weights and biases for the first layer:
    $$
   \mathbf{W}_1 := \mathbf{W}_1 - \eta \frac{\partial L}{\partial \mathbf{W}_1}
   $$
   $$
   \mathbf{b}_1 := \mathbf{b}_1 - \eta \frac{\partial L}{\partial \mathbf{b}_1}
   $$
