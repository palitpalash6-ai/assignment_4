"""
Home Assignment 4 - Part II, Q3: Scaled Dot-Product Attention
CS5720 Neural Network and Deep Learning - Fall 2026
Student: Palash Palit

Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(x):
    """Row-wise softmax. Subtracting the row max first avoids overflow and does not change the result."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """Return (attention_weights, output) for query, key and value matrices."""
    d = K.shape[-1]                      # key dimension
    scores = Q @ K.T                     # 1) dot product of Q and K transpose
    scaled = scores / np.sqrt(d)         # 2) scale by sqrt(d)
    weights = softmax(scaled)            # 3) softmax -> attention weights (each row sums to 1)
    output = weights @ V                 # 4) weighted sum of the value vectors
    return scores, scaled, weights, output


if __name__ == "__main__":
    Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    scores, scaled, weights, output = scaled_dot_product_attention(Q, K, V)
    print("Q K^T (raw scores):\n", scores)
    print(f"\nScaled scores (divided by sqrt(d) = sqrt({K.shape[-1]}) = {np.sqrt(K.shape[-1]):.0f}):\n", scaled)
    print("\n1. Attention weights (after softmax):\n", weights)
    print("   Row sums:", weights.sum(axis=1))
    print("\n2. Final output (weights x V):\n", output)
