# COMP4027
 Haskell deep learning library.

Dependencies (each of which can be installed via cabal e.g. 'cabal install random-1.2.0'):
  - random-1.2.0
  - deepseq-1.4.5.0
  - JuicyPixels-util-0.2 (for the MNIST module)
  - JuicyPixels-util-3.3.5 (for the MNIST module)
  
Acknowledgements:
  - The MNIST module and subdirectory are NOT my work, they have been sourced from https://git.thm.de/akwl20/neural-network-haskell

Using the code:
  - Hyper-parameters for the learning algorithms are located in 'MLP_utils': these include learning rate, batch size, alpha value (i.e. impact of momentum) and gradient clipping.
    - Currently, parallelization is only supported for 2 threads, so to prevent nThreads from being set to <1 or >2 the useParallelism function can either be set to True (to partition data for 2 threads) or False (to not partition the data).  
  - Run 'MLP_proof_of_concept' to train and test a model (with 2 input, 2 hidden and 1 output neurons) on the XOR gate function.
    - The model trained can either be randomly initialized or pre-initialized, by switching the commented/uncommented definitions for main.
    - The model can either be trained using online learning or batch learning, by switching the commented/uncommented definitions for ((tE,_),model') in trainXORForN.
    - The number of epochs to train for is the initial state passed to trainXORForN in trainXOR (the second argument of evalState).
  - Run 'MLP_training_MNIST' to train the pre-initialized model located in 'initModel' on the MNIST dataset.
    - The number of samples to train with and what number to use for validation are defined as tSize and vSize in trainMNIST.
    - To toggle stochastic gradient descent, switch the commented/uncommented bindings for dataset in trainMNIST.
    - The model can either be trained using online learning or batch learning, by switching the commented/uncommented definitions for ((tE,vE),model') in trainNetForN.
    - The number of epochs to train for is the definition of nEpochs in trainMNIST.
    - The trained model can either be printed or strictly/fully evaluated, by switching the commented/uncommented lines in timedRun.
      - To observe the time a model takes to train, it is better to fully evaluate it without printing so that printing time is not measured.
      - To test a model, it must be printed so that it can be copied and pasted into the definition of getMNISTmodel in 'TrainedModel', where activation functions must manually be inserted before this file is saved.
  - Run 'MLP_testing_MNIST' to test the trained model located in 'TrainedModel' on the MNIST dataset.
    - Model predictions are compared to testset labels to show how many samples (out of 10000) the model correctly predicts. 

Comments:
  - If an error occurs due to lack of virtual memory, then you may need to increase the paging file size due to the size of the models used to learn the MNIST dataset.
    - On Windows, this can be done by going into system properties (type 'edit the system environment variables' into the task bar); advanced; performance options; advanced; virtual memory.
  - Examples of pre-trained models can be found in the models.txt file, along with the hyper-parameters used to obtain these models.
  
