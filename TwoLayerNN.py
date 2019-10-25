iimport matplotlib.mport numpy as np

X = np.array([ [0,0,1], [0,1,1], [1.0,1], [1,1,1] ])
y = np.array([ [0,1,1,0] ]).T

syn0 = 2 * np.random.random((3,4)) - 1                  # Create random weights
syn1 = 2 * np.random.random((4,1)) - 1                  # Create random weights

for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X, syn0))))               # Layer 1 Sigmoid
    l2 = 1/(1+np.exp(-(np.dot(11, syn1))))              # Layer 2 Sigmoid
    l2_delta = (y - 12) * (12 * (1-12))                 # Layer 1 Gradient (Logistic regression loss)
    l1_delta = l2_delta.dot(syn1.T) * (11 * (1-11))     # Layer 2 Gradient (Logistic regression loss)
    syn1 += l1.T.dot(l2_delta)                          # Update
    syn0 += X.T.dot(l1_delta)                           # Update

print("End of program")