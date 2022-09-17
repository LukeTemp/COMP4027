In this module is a pre-initialized model to learn the MNIST dataset as well as all of the functions needed to randomly initialize a multi-layer perceptron model (which were also used to generate the pre-initialized model, although the correct activation functions had to be inserted manually).

> module InitModel
>     ( initWs
>     ) where

> import Types -- for UnactivatedLayer type
> import System.Random -- getStdRandom

The following 2 functions perform random weight initialization; the first function initializes a single connection between two neurons in adjacent layers, whilst the second function generates all connections needed for a network of the given architecture. For example, initWs [2,3,2] returns all connections for multi-layer perceptron with 2 input neurons, 3 hidden neurons and 2 output neurons.
Also note that initWs adds sigmoid activation to each layer: this is the default activation function for randomly initialized layers. Currently, if someone wants to change this to a different activation function (e.g. softmax) then this must be done manually. In future, there will be an option to automate this e.g. initWs may also take a list of activation functions.

> initWeight :: IO (Double, Double)
> initWeight = normRand >>= \r -> return (r, 0.0)
>
> initWs :: [Int] -> IO [UnactivatedLayer] 
> initWs xs = initWs' xs >>= \ws -> return $ zipWith3 UL (tail xs) ws (repeat Sigmoid)
>     where
>     initWs' []      = return []
>     initWs' [ol]    = return []
>     initWs' (cl:ls) = do  -- current layer excludes bias neuron; (cl+1) includes bias neuron
>         rs <- sequence . replicate (head ls) . sequence . replicate (cl+1) $ initWeight
>         rss <- initWs' ls
>         return (rs : rss)

Note that normRand returns a number from the normal distribution via the Box-Muller algorithm:

> normRand :: IO Double
> normRand = boxMull 0 1 
>
> boxMull :: Double -> Double -> IO Double
> boxMull mean dev = do
>     x <- getStdRandom $ randomR (0.0, 1.0)
>     y <- getStdRandom $ randomR (0.0, 1.0)
>     return $ mean + sqrt (-2 * log x) * cos (2 * pi * y) * dev
