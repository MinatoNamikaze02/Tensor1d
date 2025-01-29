import random
from tensor import Tensor1d

class Neuron:
    def __init__(self, nin):
        self.w = Tensor1d([random.uniform(-1, 1) for _ in range(nin)]) * 1.0
        self.b = Tensor1d([random.uniform(-1, 1)]) * 1.0

    def __call__(self, x, activation="tanh"):
        # print(f"Tensor: {x}, Has _backward: {hasattr(x, '_backward')}")
        # print(f"Tensor: {self.w}, Has _backward: {hasattr(self.w, '_backward')}")
        # print(f"Tensor: {self.b}, Has _backward: {hasattr(self.b, '_backward')}")
        if isinstance(x, list):
            out = Tensor1d([0.0])
            for x, w in zip(x, self.w):
                out += x * w
        else:
            out = x * self.w
        out = out.sum() + self.b

        # print(f"Tensor: {out}, Has _backward: {hasattr(out, '_backward')}")
        return out

    def parameters(self):
        return [self.w, self.b]  # Correct list of parameter tensors


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x, activation="tanh"):
        outputs = [neuron(x) for neuron in self.neurons]
        stacked =  Tensor1d.stack(outputs)
        # print(f"Tensor: {stacked}, Has _backward: {hasattr(stacked, '_backward')}")
        return stacked
  
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x, activation="tanh"):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
  

if __name__ == "__main__":
    x = Tensor1d([random.uniform(-1,1) for _ in range(10)]) * 1.0
    y = Tensor1d([random.uniform(-1,1) for _ in range(3)]) * 1.0

    # create a simple MLP
    mlp = MLP(10, [5, 3])

    epochs = 10
    lr = 0.01

    for epoch in range(epochs):
        # forward pass
        pred = mlp(x)
        # print(f"Prediction: {pred}")

        # backward pass
        loss = (pred - y).sum()
        loss._backward()

        # update weights
        for p in mlp.parameters():
            grad = p.get_grad()
            print(f"Parameter: {p}, Gradient: {grad}")
            p -= p.get_grad() * lr

        print(f"Epoch: {epoch}, Loss: {loss[0]}")
        print("")

