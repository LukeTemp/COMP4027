# COMP4027
 Haskell deep learning library.

Dependencies:
  - random-1.2.0: can be installed via cabal with the command 'cabal install random-1.2.0'

Using the code (GHCI):
  - Run 'testF' to check the correctness of training.
  - Run 'main' to check the efficiency of training.
  - Both of the above functions will:
    - Initialize the model; currently initializes to a fixed preset 'initF' but can be changed to randomly initialized weights by binding 'initWs $ map (length . weights) initF' in 'trainF'.
    - Train the model on the target function given by zipping the inputs and targets in 'trainF'.
  - After training the model: 
    - 'testF' prints the weights as well as computing and printing model predictions for each input in the test data (currently defined in the where clause of 'testF').
    - 'main' prints convergence time excluding IO operations beyond getCurrentTime (note that if weights are randomly initialized, this IO time will be included in the measured time).

Comments:
  - Currently, the sequential multi-layer perceptron is trained and tested on a custom binary function described on line 138 of 'MLP_sequential.lhs':
    - Since this function is just a boolean gate scaled over 5 dimensions, generalization is neither necessary nor practical so the test data is the same as the training data.