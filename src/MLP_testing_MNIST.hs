module MLP_testing_MNIST where

{- |
  In this module are a number of functions that have been used to test models
  that have been trained on the MNIST dataset.

  The functions below have been defined to test models that have been trained
  on the MNIST dataset. The testMNIST function obtains the MNIST test dataset
  and uses this to test the model. In testModel, each sample is tested to see
  if the one hot encoded prediction is the same as the target, if so then 1
  is added to a sum, otherwise 0 is added. The final sum tells us how many
  samples (out of 10000) the model predicted correctly.
-}

import Data.Bool (bool)
import Data.Tuple.Extra (both)
import Data.Time.Clock -- clock to measure convergence time
import Types -- all custom types/classes for parallel multi-layer perceptrons
import MNIST -- MNIST dataset for target function. Source: https://git.thm.de/akwl20/neural-network-haskell
import Data.Matrix as Matrix -- toList operation for matrices
import MLP_utils -- for feedFrwd and toLoss
import qualified Data.Text.IO as TXTIO
import qualified Data.Text as TXT

-- Returns bool indicating whether the sample was predicted correctly.
testSample :: [UnactivatedLayer] -> ([Double],[Double]) -> Bool
testSample model (input,target) =
  target == (predictClass . outputs . head $ feedFrwd model input)
  where
  predictClass :: [Double] -> [Double]
  predictClass xs = [if x == maximum xs then 1.0 else 0.0 | x <- xs]
  -- ^ Converts a softmax output to a one hot vector prediction.

-- | Returns the number of samples predicted correctly.
testModel :: [UnactivatedLayer] -> [([Double], [Double])] -> Int
testModel model testData = sum $ map (bool 0 1 . testSample model) testData

-- | Measures the time taken to test a model that has been trained on the MNIST
-- dataset. Also prints how many samples out of the 10000-sample MNIST test-set
-- the model predicted correctly.
timedRun :: IO ()
timedRun = do
  let preprocess :: (Real a, Fractional b) => [(Matrix a, Matrix a)] -> [([b], [b])]
      preprocess = map $ both $ map realToFrac . Matrix.toList
  model <- read . TXT.unpack <$> TXTIO.readFile "trained_model" :: IO [UnactivatedLayer]
  testData <- preprocess <$> getTestSamples
  t0 <- getCurrentTime
  let numTruePositives = testModel model testData
  -- Force evaluation by printing the # of true positives before taking the time:
  putStrLn ("Accuracy: " ++ show numTruePositives ++ " out of " ++ show (length testData))
  t1 <- getCurrentTime
  putStrLn ("Time to test: " ++ show (diffUTCTime t1 t0))
