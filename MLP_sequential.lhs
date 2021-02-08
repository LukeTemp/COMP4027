Import: clock to measure convergence time; getStdRandom for random weight initialization; transpose to accumulate the error passed back to neurons in the previous layer; all custom types/classes for multi-layer perceptrons; deepseq to fully evaluate the trained network (to compare convergence times between alternative/future implementations):

> import Data.Time.Clock
> import System.Random
> import Data.List 
> import Types
> import Control.DeepSeq

The next function, nextStage, captures a common pattern used when feeding forward and back propagating through the network (see the definitions of feedFrwd and backProp).
The first argument, (l -> [NrnSignal] -> l'), is a function which takes an unprocessed layer l and signals (e.g. outputs or error) from the last layer that was processed, and returns the result of processing layer l.
The second argument, [l], represents the network at its previous stage of evaluation (a list of unprocessed layers). When nextStage is used to feed forward, the layers have not yet been activated so [l] :: [UnactivatedLayer]; when nextStage is used to back propagate, the layers have already been activated so [l] :: [ActivatedLayer].
The third argument, [NrnSignal], is the initial signal for the next stage of evaluation (either network inputs or prediction error). In the definition of the function, foldl is used to integrate list reversal with layer processing, effectively turning the accumulator argument into a stack so that state from the previous processed layer can be taken efficiently from the head.
When nextStage is used to feed forward, [(b,c)] :: [ActivatedLayer], otherwise it is used to back propagate so [(b,c)] :: [BackpropLayer]. Put simply, nextStage evaluates the network by processing it and reversing it so that it is ready for the next stage of evaluation:

> nextStage :: (Layer l, PropagatingLayer l') => (l -> [NrnSignal] -> l') -> [l] -> [NrnSignal] -> [l'] 
> nextStage f (x:xs) s = foldl propagate [f x s] xs
>     where propagate zs@(y:ys) x = f x (signals y) : zs

By using nextStage it is easy to define the feed forward function and back propagation function. Each function need only pass the necessary arguments to the nextStage function: a function to process layers, the unprocessed network and a list of initial signals to propagate through the network:

> feedFrwd :: [UnactivatedLayer] -> [NrnSignal] -> [ActivatedLayer]
> feedFrwd uls inps = nextStage activLay uls inps
>    
> backProp :: [ActivatedLayer] -> [NrnSignal] -> [BackpropLayer]
> backProp als err = nextStage backpropLay als err 

Layers are activated using the activLay function, which is given an unactivated layer and previous layer outputs so that it can activate each neuron in the layer. Neurons are activated in the activNrn function, which takes the weights that connect a neuron to the previous layer and multiplies them by the corresponding outputs from the previous layer.

> activLay :: UnactivatedLayer -> [NrnSignal] -> ActivatedLayer
> activLay (UL wss) prev = AL wss prev $ map (sigf . sum . activNrn prev) wss
>
> activNrn :: [NrnSignal] -> NrnWeight -> [NrnSignal]
> activNrn prev ws = zipWith (*) (1.0 : prev) (map fst ws)          

The sigf function implements sigmoid activation for neurons during a forward pass:

> sigf :: Double -> Double
> sigf x = 1 / (1 + e ** (-x))
>     where e = 2.718281828459045
                                
Error is propagated from one layer to the previous layer using the backpropLay function; this is done by propagating error from each neuron in the current layer to the previous layer using backNrn. The backNrn function is given the output, error, weights and inputs for a neuron n; the output and error of n is used to compute the error gradient, which is used (along with the weights and inputs of n) to update the connections between n and the previous layer. Each connection is updated by applying the weight and momentum update equation defined in appEquation, and given an error value (input neuron weight * output neuron gradient). To fully propagate error to the previous layer, each neuron in the previous layer must be given the sum of all errors that have been propagated through connections to the current layer; this is handled by accumError (in the where clause of backpropLay): 

> backpropLay :: ActivatedLayer -> [NrnSignal] -> BackpropLayer
> backpropLay (AL wss is os) err = (uncurry BL) . accumError . unzip $ map (backNrn is) (zip3 os err wss) 
>     where accumError (wss,ess) = (wss, tail . accum $ ess) -- tail ensures that error is not propagated to a bias neuron.
>
> backNrn :: [Double] -> (Double,NrnSignal,NrnWeight) -> (NrnWeight,[NrnSignal])
> backNrn is (o,e,ws) = (zipWith (appEquation g) (1.0 : is) ws, map ((*g) . fst) ws) -- (1.0 : is) ensures that bias neuron weights are updated. 
>     where g = gClip $ grad e o
> 
> appEquation :: Double -> Double -> (Double, Double) -> (Double, Double)
> appEquation g i (w,dw) =  (w+dw', dw')
>     where dw' = dw * getAlpha + (i * g * getEta)

Note that gClip implements gradient clipping to prevent gradient explosion, which can occur when using sigmoid activation to learn non-linear functions.

> gClip :: Double -> Double
> gClip g | g > 0.1             = 0.1
>                | g < (-0.1)       = (-0.1)
>                | otherwise = g

The accum function adds elements from the same index of each sublist in a given list; each sublist represents a list of errors propagated from a neuron in layer n to its input neurons in layer n-1, so the error for a neuron in layer n-1 is the sum of errors propagated back to it from the neurons that it outputs to in layer n. This operation is equivalent to summing over the columns of a matrix:

> accum :: [[NrnSignal]] -> [NrnSignal]
> accum xs = [sum x | x <- transpose xs]

The derivative of sigmoid and gradient computation functions are defined below. In the future, the AD library will be used to exploit automatic differentiation so that the derivatives of sigf are computed whilst evaluating it, rather than using a separate function derSig to compute derivatives after sigf has been evaluated:

> derSig :: Double -> Double
> derSig x = x * (1.0 - x)
>
> grad :: Double -> Double -> Double
> grad e o = e * derSig o             

Note that getEta and getAlpha define values for the learning rate (the magnitude of a weight update) and alpha (the impact of momentum on a weight update), respectively:

> getEta :: Double
> getEta = 0.1
> getAlpha :: Double
> getAlpha = 0.25

With forward passing and back propagation implemented, it is now possible to combine these in a function that trains the model on a given training sample. If a model should use batch learning (not yet implemented) then back propagation should occur after n forward passes (each on a different training sample), using the summed or mean error of all n predictions. If a model should use online learning, then back propagation should occur after every forward pass. Since a sample is just one element of a dataset, and the accuracy of a model is judged on how it performs on a dataset, the loss of a training sample should be added to the loss of the epoch. The function below, evalSample, implements online learning in accordance with these specifications; e is the error of an epoch and map fst is applied to a back propagated network to extract the updated parameters: 

> evalSample ::  (Double, [UnactivatedLayer]) -> ([Double],[Double]) -> (Double, [UnactivatedLayer])
> evalSample (e,model) (input,target) =  (e + toLoss es, model')
>     where 
>     all_outs = feedFrwd model input
>     model' = map (UL . weights) (backProp all_outs es)
>     es = zipWith (-) target (outputs . head $ all_outs) -- extracts the output layer's output (i.e. the model's prediction) and computes errors from the target

Note that toLoss implements the Mean Squared Error loss function:

> toLoss :: [Double] -> Double
> toLoss es = sqrt . average $ map (**2) es
>
> average :: Fractional a => [a] -> a
> average xs = sm / ln 
>     where (sm, ln) = foldr (\x (y, i) -> (x+y, i+1)) (0,0) xs 

Next, the program requires a function that can run the model for an epoch, and a function that can repeatedly run epochs until epoch error has been suitably minimized. The first function can be defined concisely as a fold which evaluates each sample in a given dataset, accumulating the total error for the epoch whilst updating the network parameters. The second function must repeatedly run epochs until termination criteria is met; the current termination criteria is to check the error after each epoch, if it is less than a threshold (0.2) then the weights will be kept, otherwise another epoch will be run:

> evalEpoch :: [UnactivatedLayer] -> [([Double],[Double])] -> (Double, [UnactivatedLayer])
> evalEpoch model trainData = foldl evalSample (0, model) trainData
>
> trainNet :: [[Double]] -> [[Double]] -> [UnactivatedLayer] -> [UnactivatedLayer]
> trainNet allInps allTargs model = if fst epoch < 0.5 then model else trainNet allInps allTargs (snd epoch)
>     where epoch = evalEpoch model (zip allInps allTargs) 

All the tools needed to train an initialized logistic multi-layer perceptron have now been implemented; the next step is to implement the functions needed for initialization. The following 2 functions perform random weight initialization; the first function initializes a single connection between two neurons in adjacent layers, whilst the second function generates all connections needed for a network of the given architecture. For example, initWs [2,3,2] returns all connections for an ANN with 2 input neurons, 3 hidden neurons and 2 output neurons:

> initWeight :: IO (Double, Double)
> initWeight = do
>                         r <- normRand
>                         return (r, 0.0)
>
> initWs :: [Int] -> IO [UnactivatedLayer] -- USE replicateM to simplify this definition!!!
> initWs xs = fmap (map UL) $ initWs' xs
>     where  
>     initWs' []       = return []
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
>                    x <- getStdRandom $ randomR (0.0, 1.0)
>                    y <- getStdRandom $ randomR (0.0, 1.0)
>                    return $ mean + sqrt (-2 * log x) * cos (2 * pi * y) * dev                   

Now that everything needed to initialize and train multi-layer perceptrons has been implemented, it is time to test the code. An initialized model is defined by initF, so that we can train models deterministically from the same initial parameters when we compare sequential backpropagation with parallel backpropagation (WIP). The trainF function is used to train the model, whilst the testF function is used to display the results of testing the trained model.
The learned function below is an XOR gate scaled over 5 dimensions: a 5d vector [i1,i2,i3,i4,i5] is input and a 4d vector [i1 XOR i2, i2 XOR i3, i3 XOR i4, i4 XOR i5] is output. Since XOR is used to obtain every output element, this function preserves non-linearity between each set of inputs that map to an output. This function will be learned, rather than a typical XOR gate, because it is more complex so training will take longer and any speedup achieved by parallel backpropagation (WIP) will be more clear.

> initF :: [UnactivatedLayer] -- currently configured to a [5,7,4] architecture i.e. 5 input neurons, 7 hidden neurons and 4 output neurons.
> initF = [UL [[(-0.12449641112336,0.0),(1.6437777799671236,0.0),(-0.9091490654100092,0.0),(1.5419569387761579e-2,0.0),(-1.362255561951286,0.0),(-0.5150749094688205,0.0)],[(1.0201590976472137,0.0),(0.7579756624450967,0.0),(0.11054293854888786,0.0),(0.10237979479728661,0.0),(-0.6311808878411262,0.0),(-1.2596536578413917,0.0)],[(-1.6192504119764695,0.0),(0.5015542491294096,0.0),(0.17104984961130845,0.0),(0.18522334130565993,0.0),(0.8280523627989466,0.0),(1.7457301936399718,0.0)],[(-0.921654078228532,0.0),(-1.4161644407462615,0.0),(0.98837926765552,0.0),(-0.5222541806914169,0.0),(1.4871441939106314,0.0),(3.134980381447465e-2,0.0)],[(-1.39571759187251,0.0),(-0.24135573733738233,0.0),(-0.24474618875032944,0.0),(3.716261282160874e-2,0.0),(1.2874044134055218,0.0),(-0.6153758264934315,0.0)],[(-0.3453606105445722,0.0),(0.29333972813475767,0.0),(0.353462168736623,0.0),(-1.4872799254871023,0.0),(-0.5513090295835475,0.0),(0.8995237226678555,0.0)],[(2.0501618009773255,0.0),(-0.1738290265865568,0.0),(1.5636576817493746,0.0),(0.5474597079881128,0.0),(-1.957084038882988,0.0),(-0.5845497525175318,0.0)]],UL [[(1.0115474366197204,0.0),(-0.31502425536005324,0.0),(-0.10403809309978268,0.0),(6.199923002722395e-3,0.0),(-0.15975545204744238,0.0),(-1.5825453909008007,0.0),(0.5210302928535288,0.0),(1.6500225480951027,0.0)],[(-1.123569668309377,0.0),(-0.5032514641728917,0.0),(0.16648303891436295,0.0),(0.6107325149963377,0.0),(5.039008667297707e-2,0.0),(-7.3561359593401305e-3,0.0),(-0.6627501244556316,0.0),(-0.6739899644665761,0.0)],[(-0.9098309481528618,0.0),(-1.4944112578868312,0.0),(1.6055614631578523,0.0),(1.869684722892049,0.0),(-1.62431304213004,0.0),(0.6295053707446395,0.0),(1.845236163102067,0.0),(-1.284045179475841,0.0)],[(1.640057573200443,0.0),(-0.8841824776676669,0.0),(0.4822137927654142,0.0),(0.719703030386541,0.0),(0.17037610188843103,0.0),(1.5478843170035184,0.0),(1.224336249803835,0.0),(0.764524015006373,0.0)]]]
>
> trainF :: [UnactivatedLayer]
> trainF = trainNet inps targs initF 
>         where 
>         inps  = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,0],[0,0,0,1,1],[0,0,1,0,0],[0,0,1,0,1],[0,0,1,1,0],[0,0,1,1,1],[0,1,0,0,0],[0,1,0,0,1],[0,1,0,1,0],[0,1,0,1,1],[0,1,1,0,0],[0,1,1,0,1],[0,1,1,1,0],[0,1,1,1,1],[1,0,0,0,0],[1,0,0,0,1],[1,0,0,1,0],[1,0,0,1,1],[1,0,1,0,0],[1,0,1,0,1],[1,0,1,1,0],[1,0,1,1,1],[1,1,0,0,0],[1,1,0,0,1],[1,1,0,1,0],[1,1,0,1,1],[1,1,1,0,0],[1,1,1,0,1],[1,1,1,1,0],[1,1,1,1,1]]
>         targs = [[0,0,0,0],  [0,0,0,1],  [0,0,1,1],  [0,0,1,0],  [0,1,1,0],  [0,1,1,1],  [0,1,0,1],  [0,1,0,0],  [1,1,0,0],  [1,1,0,1],  [1,1,1,1],  [1,1,1,0],  [1,0,1,0],  [1,0,1,1],  [1,0,0,1],  [1,0,0,0],  [1,0,0,0],  [1,0,0,1],  [1,0,1,1],  [1,0,1,0],  [1,1,1,0],  [1,1,1,1],  [1,1,0,1],  [1,1,0,0],  [0,1,0,0],  [0,1,0,1],  [0,1,1,1],  [0,1,1,0],  [0,0,1,0],  [0,0,1,1],  [0,0,0,1],  [0,0,0,0]]
>
> testF :: IO ()
> testF = do
>     putStrLn ("weights: " ++ show model)
>     mapM_ print [show (map floor is) ++ " : " ++ show (mround . outputs . head $ feedFrwd model is) | is <- inps]
>         where 
>         model  = trainF
>         mround = map round
>         inps   = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,0],[0,0,0,1,1],[0,0,1,0,0],[0,0,1,0,1],[0,0,1,1,0],[0,0,1,1,1],[0,1,0,0,0],[0,1,0,0,1],[0,1,0,1,0],[0,1,0,1,1],[0,1,1,0,0],[0,1,1,0,1],[0,1,1,1,0],[0,1,1,1,1],[1,0,0,0,0],[1,0,0,0,1],[1,0,0,1,0],[1,0,0,1,1],[1,0,1,0,0],[1,0,1,0,1],[1,0,1,1,0],[1,0,1,1,1],[1,1,0,0,0],[1,1,0,0,1],[1,1,0,1,0],[1,1,0,1,1],[1,1,1,0,0],[1,1,1,0,1],[1,1,1,1,0],[1,1,1,1,1]]

The main function below measures the time taken for the model to converge. To ensure that the network is trained it is fully evaluated using deepseq:

> main = do 
>     t0 <- getCurrentTime
>     deepseq trainF $ return ()
>     t1 <- getCurrentTime
>     putStrLn ""
>     putStrLn ("Time to converge: " ++ show (diffUTCTime t1 t0))

To use deepseq on the trained network, an instance of NFData must be defined for the UnactivatedLayer type:

> instance NFData UnactivatedLayer where
>     rnf (UL ws) = rnf ws