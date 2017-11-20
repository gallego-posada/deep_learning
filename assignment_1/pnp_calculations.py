import numpy as np

def relu(x):
    return np.maximum(0,x)

def der_relu(x):
    return 1.0 * (x>0)

def loss(yout, ygt):
    N = yout.shape[-1]
    return np.linalg.norm(yout - ygt)**2/(2.0*N)

LEARNING_RATE = 0.5
EPOCHS = 1

X = np.array([[0.75, 0.2, -0.75, 0.2],[0.8, 0.05, 0.8, -0.05]])
Ygt = np.array([[1, 1, -1, -1]])
N = Ygt.shape[-1]
W1 = np.array([[0.6, 0.01],[0.7, 0.43],[0, 0.88]])
W2 = np.array([[0.02, 0.03, 0.09]])

def forward(X,W1,W2):
    s1 = np.dot(W1,X)
    z1 = relu(s1)
    s2 = np.dot(W2, z1)
    z2 = s2

    return s1, z1, s2, z2

for epoch in range(EPOCHS):

    s1, z1, s2, z2 = forward(X,W1,W2)


    print(s1)
    print(z1)
    print(s2)
    print(z2)

    L = loss(z2, Ygt)
    print("Loss", L)

    delta2 = (z2 - Ygt)/N
    grad_W2 = delta2.dot(z1.T)

    delta1 = np.dot(W2.T,delta2) * der_relu(s1)
    grad_W1 = delta1.dot(X.T)

    W1 -= LEARNING_RATE * grad_W1
    W2 -= LEARNING_RATE * grad_W2


    print(delta1)
    print(grad_W1)
    print(delta2)
    print(grad_W2)


print("Final params")
print(W1)
print(W2)

s1, z1, s2, z2 = forward(X,W1,W2)

L = loss(z2, Ygt)
print("Loss", L)


# print(s1)
# print(z1)
# print(s2)
# print(z2)

# print(delta2)
# print(W2.T)
# print(np.dot(W2.T,delta2))
