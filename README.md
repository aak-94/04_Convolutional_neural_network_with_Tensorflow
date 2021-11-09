# Convolutional_neural_network_with_Tensorflow
 Convolutional_neural_network_with_Tensorflow
  ## Goal:
 - To understand different APIs for neural network developement in tensorflow framework
 
 ## APIs types
1. Sequential API:
 It is ideal for building models where each layer has exactly one input tensor and one output tensor. Uusing the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks.
Note:  The use of the loss calculation method and the type of labels in the input dataset. 
2. Functional API:
The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the
Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line,
a Functional model is a graph, where the nodes of the layers can connect in many more ways than one.

## About the code:
The code is a straightforward implementation of the neural network for the digit classification.
Dataset used = MNIST dataset
In the main function, there is an option to select the API. It's a modular code, thus helps to understand the stepwise implementation of the APIs.
Start with the main function and refer to the indivdual functions in sequence for better understanding.

## References:
- https://keras.io/api/losses/
- https://victorzhou.com/blog/keras-cnn-tutorial/
