# General Terms
## Epochs
- one epoch = one forward pass and one backward pass of all the training examples

## Batch Size
- The batch size defines the number of samples that will be propagated through the network simultaneously. 
- The number of samples is often a power of 2, to facilitate memory allocation on GPU.
- The algorithm takes the first 128 samples (from 1st to 128th) from the training dataset and trains the network. Next it takes the second 128 samples (from 129st to ...th) and trains the network again.

## Activation Functions
Neural network activation functions are a crucial component of deep learning. Activation functions determine the output of a deep learning model, its accuracy, and also the computational efficiency of training a model. Activation functions also have a major effect on the neural network’s ability to converge and the convergence speed, or in some cases, activation functions might prevent neural networks from converging in the first place. <br>
The function is attached to each neuron in the network, and determines whether it should be activated (“fired”) or not, based on whether each neuron’s input is relevant for the model’s prediction. <br>
Modern neural network models use non-linear activation functions. They allow the model to create complex (non-linear) mappings between the network’s inputs and outputs. 

### Identity (Linear)
- This function is linear. Therefore, the output of the functions will not be confined between any range.
- Not possible to use for backpropagation because the derivative of the function is a constant, and has no relation to the input. 
- Range: (-infinity to infinity)

### Binary Step
- A binary step function is a threshold-based activation function. If the input value is above or below a certain threshold, the neuron is activated and sends exactly the same signal to the next layer.
- The problem with a step function is that it does not allow multi-value outputs—for example, it cannot support classifying the inputs into one of several categories.
- Range: 0 to 1

### Sigmoid
- The sigmoid function is especially used for models where we have to predict the probability as an output.
- Range: 0 to 1
- Advantages:
   - Smooth gradient
   - Output values bound between 0 and 1
- Disadvantages:
   - Vanishing gradient—for very high or very low values of X, there is almost no change to the prediction, causing a vanishing gradient problem. 
   - Outputs not zero centered.
   - Computationally expensive

### Tanh
- The tanh function is mainly used classification between two classes.
- Range: -1 to 1
- Advantages:
   - Zero centered—making it easier to model inputs that have strongly negative, neutral, and strongly positive values.
   - Like Sigmoid
- Disadvantages:
   - Like Sigmoid

### Softmax Function
- Typically Softmax is used only for the output layer, for neural networks that need to classify inputs into multiple categories.
- Advantages:
   - Able to handle multiple classes

### ReLU
- Often used as default activation function
- Range: 0 to infinity
- Advantages:
   - Computationally efficient (allows the network to converge very quickly)
   - Although it looks like a linear function, ReLU has a derivative function and allows for backpropagation
   - It is capable of outputting a true zero value.
- Disadvantages:
   - The Dying ReLU problem (when inputs approach zero, or are negative, the gradient of the function becomes zero, the network cannot perform backpropagation and cannot learn)

### Leaky ReLU
- Range: xxxx to infinity
- Advantages:
   - Prevents dying ReLU problem (this variation of ReLU has a small positive slope in the negative area, so it does enable backpropagation, even for negative input values)
   - Like ReLU
- Disadvantages:
   - Results not consistent (leaky ReLU does not provide consistent predictions for negative input values).

### Logistic



![](./pictures/activation_functions.png)