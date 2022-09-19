> {-# LANGUAGE LambdaCase #-}
>
> module MLP_training_MNIST where

In this module are a number of functions that have been used to train models from and test the code in MLP_utils using the MNIST dataset.

> import Data.Time.Clock -- clock to measure convergence time
> import MNIST -- MNIST dataset for target function. Source: https://git.thm.de/akwl20/neural-network-haskell
> import SGDF -- Stochastic Gradient Descent Functions: shuffle function is needed to implement Stochastic Gradient Descent
> import InitModel -- the initMNIST function provides a fixed randomly initialised model to train on the MNIST dataset
> import MLP_utils -- for evalEpoch, evalEpochB and batchSize
> import Types -- all custom types/classes for parallel multi-layer perceptrons
> import Control.Monad.State.Lazy -- State Monad for implicit counting
> import Data.Matrix -- toList operation for matrices
> import System.Random -- getStdRandom

> import Control.DeepSeq -- deepseq to fully evaluate the trained network so that convergence times can be compared.

> import qualified Data.Text.IO as TXTIO
> import qualified Data.Text as TXT

To test the batch learning code, I created and used the functions below. This test simply proves that when using a batch size of 1 the batch learning code performs the same as the 
online learning code - it does not necessarily prove that the batch learning code is correct. To run the test, the main function should be redefined as 'main = test initMNIST >>= print' 
and in MLP_utils.lhs batchSize should be redefined as 'batchSize = 1'. The functions tTSize and tVSize should be set to the total number of samples to use and what number of these 
should be used for validation, respectively (note that the total number of samples minus the number of validation samples gives you the number of training samples). 
Note that getTrainingSamples uses the imported MNIST module to get the MNIST training dataset and shuffle uses the SGDF module to randomly reorganise the dataset before taking data 
samples from it. 

> tTSize = 100 -- test total size
> tVSize = 10 -- test validation size
>
> test' :: [UnactivatedLayer] -> ([([Double], [Double])], [Bool]) -> IO Bool
> test' model tvData = do
>     let ((tE1,vE1),m1) = evalEpochBatch model tvData
>     let ((tE2,vE2),m2) = evalEpochOnline model tvData
>     print $ (tE1 / fromIntegral tTSize, vE1 / fromIntegral tVSize)
>     print $ (tE2 / fromIntegral tTSize, vE2 / fromIntegral tVSize)
>     return (map weights m1 == map weights m2 && map size m1 == map size m2)
>
> test :: [UnactivatedLayer] -> IO Bool
> test model = do
>     dataset <- shuffle . take tTSize <$> getTrainingSamples -- (take 100) <$> getTrainingSamples -- length <$> getTrainingSamples == 60000 
>     let validation   = valids (tVSize, (tTSize-tVSize))
>     let format (x,l) = (realToFrac <$> toList x, realToFrac <$> toList l)
>     let dataset'     = format <$> dataset
>     test' model (dataset',validation)

To train a model on the MNIST dataset, I created the function below: trainNetForN trains a network for n epochs (where n is the initial state passed to this function). 
On simpler functions I trained the model until the error of an epoch was below an error threshold, but since the MNIST target function is much more complex, it will take longer to 
learn and we do not know how many epochs will cause the model to underfit/overfit to the training data. Therefore, this function gives us more control over the training process.

> trainNetForN ::
>     [UnactivatedLayer]
>   -> ([([Double], [Double])], [Bool])
>   -> Int
>   -> IO ((Double, Double), [UnactivatedLayer])
> trainNetForN model (trData,valid) = \case
>   0 -> error "cannot train a model for 0 epochs" -- replace with a Left
>   1 -> evalEpochFunction model (trData,valid)
>   nEpochs -> do
>     (_, model') <- evalEpochFunction model (trData,valid)
>     trainNetForN model' (trData, valid) $ nEpochs - 1
>   where evalEpochFunction = evalEpochOnlineWithPrinting -- online learning
>   --where evalEpochFunction = evalEpochBatch -- batch learning

To handle data preprocessing, such as shuffling the dataset for SGDF and formatting matrices into lists, I defined the trainMNIST function below. This function obtains the MNIST
training dataset, preprocesses it and feeds it to the trainNetForN function along with a specified number of epochs (currently set at 1). The resulting errors are then divided by
the number of training/validation samples over the batchSize to normalize them. Note that if a larger number of epochs were used, e.g. x epochs, then we would multiply the number
of training/validation samples by x when normalizing these errors.

> trainMNIST :: [UnactivatedLayer] -> IO ((Double, Double), [UnactivatedLayer])
> trainMNIST model = do
>     let tSize = 60000 -- out of 60000
>     let vSize = 0 -- less than tSize
>     let nEpochs = 1
>     dataset <- take tSize <$> getTrainingSamples           -- non-stochastic gradient descent
>     --dataset <- shuffle . take tSize <$> getTrainingSamples -- stochastic gradient descent
>     let validation   = valids (vSize, (tSize-vSize))     
>     let format (x,l) = (realToFrac <$> toList x, realToFrac <$> toList l)
>     let dataset'     = format <$> dataset
>     ((tE,vE),model') <- trainNetForN model (dataset', validation) nEpochs
>     let fI = fromIntegral
>     return ((tE / fI (tSize-vSize) / fI nEpochs, vE / fI vSize / fI nEpochs), model')

To assign validation booleans to each sample in the training dataset, the following 2 functions have been defined. If SGD is used to train the model then the valids function can be 
used, as this will simply use the last n samples in the shuffled training data as validation data. If SGD is not used then the user may want to randomly select their validation data,
which can done using the validsR function - this will shuffle the order of the validation booleans.

> valids :: (Int, Int) -> [Bool]
> valids (n,m) =  replicate m False ++ replicate n True

> validsR :: (Int, Int) -> [Bool]
> validsR = shuffle . valids

The timedRun function below measures training time for the model and can print this along with the training error, validation error and the model.

> timedRun :: IO ()
> timedRun = do
>     t0 <- getCurrentTime
>     initMNIST <- initWs [784,32,10]
>     ((tE,vE),model) <- trainMNIST initMNIST
>     TXTIO.writeFile "trained_model" (TXT.pack $ show model)
>     --deepseq model $ return ()   -- do not print the model
>     t1 <- getCurrentTime
>     putStrLn ("Training error: " ++ show tE)
>     putStrLn ("Validation error: " ++ show vE)
>     putStrLn ("Time to converge: " ++ show (diffUTCTime t1 t0))

The main function below is configured to train a model on the MNIST dataset.

> main :: IO ()
> main = timedRun
