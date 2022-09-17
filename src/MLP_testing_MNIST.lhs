> module MLP_testing_MNIST where

In this module are a number of functions that have been used to test models that have been trained on the MNIST dataset.

> import Data.Time.Clock -- clock to measure convergence time
> import Types -- all custom types/classes for parallel multi-layer perceptrons
> import MNIST -- MNIST dataset for target function. Source: https://git.thm.de/akwl20/neural-network-haskell
> import Data.Matrix -- toList operation for matrices
> import MLP_utils -- for feedFrwd and toLoss

> import qualified Data.Text.IO as TXTIO
> import qualified Data.Text as TXT

The functions below have been defined to test models that have been trained on the MNIST dataset. The testMNIST function obtains the MNIST test dataset and uses this to test the model.
In testModel, each sample is tested to see if the one hot encoded prediction is the same as the target, if so then 1 is added to a sum, otherwise 0 is added. The final sum tells us
how many samples (out of 10000) the model predicted correctly. 

> testSample :: [UnactivatedLayer] -> ([Double],[Double]) -> Bool
> testSample model (input,target) = true_pos
>     where 
>     all_outs = feedFrwd model input
>     true_pos = target == (toPred . outputs . head $ all_outs)

> testModel :: [UnactivatedLayer] -> [([Double], [Double])] -> Int
> testModel model testData = sum $ map (toInt . testSample model) testData
>     where toInt x = case x of
>                          False -> 0
>                          True -> 1

> testMNIST :: [UnactivatedLayer] -> IO Int
> testMNIST model = do 
>     testData <- getTestSamples
>     let format (x,l) = (realToFrac <$> toList x, realToFrac <$> toList l)
>     let testData' = format <$> testData
>     return $ testModel model testData'

The toPred function converts a softmax output to a one hot vector prediction.

> toPred :: [Double] -> [Double]
> toPred xs = [if x == maximum xs then 1.0 else 0.0 | x <- xs]

The timedRun function below measures testing time for the model and will print how many samples out of 10000 the model predicted correctly.

> timedRun = do 
>     t0 <- getCurrentTime
>     model <- read . TXT.unpack <$> TXTIO.readFile "trained_model" :: IO [UnactivatedLayer]
>     err <- testMNIST model
>     testData <- getTestSamples
>     putStrLn ("Accuracy: " ++ show err ++ " out of " ++ show (length testData))
>     t1 <- getCurrentTime
>     putStrLn ("Time to test: " ++ show (diffUTCTime t1 t0))

The main function below is configured to test a model trained on the MNIST dataset.

> main :: IO ()
> main = timedRun
