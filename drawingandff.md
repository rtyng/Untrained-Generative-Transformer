# Feed forward NN is applied to each position in the sequence independently

# Step 1: The Linear Transformation

The linear transformation is used to project each input into a higher dimensional space and can be represented mathematically as:

# FFN_1 = W_1 * X + b_1

W_1: weight matrix
b_1: the bias vector
X: input at a particular position

# Step 2: Non-linear Activation

FFN_1 (the output of the first step), is then passed through either a ReLU or variant (ex: GELU) function. This allows the model to understand more complex relationships by introducing non-linearity.

# FFN_2 = max(0, FFN_1) (ReLU example)


# Step 3: The second linear transformation

In the final step, the output of the second step is projected back to the original dimension through another linear transformation.

# FFN_output = W_2 * FFN_2 + b_2


Basic ASCII drawing of the model and its function
+----------------------------------+
|           Input Sequence         |
+----------------------------------+
               |  |
               |  v
+----------------------------------+
|          Positional Encoding      |
+----------------------------------+
               |  |
               |  v
+----------------------------------+
|           Encoder Layers          |
+----------------------------------+
               |  |
               |  v
+----------------------------------+
|           Decoder Layers          |
+----------------------------------+
               |  |
               |  v
+----------------------------------+
|            Output Sequence        |
+----------------------------------+
