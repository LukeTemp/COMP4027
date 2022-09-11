In this module are a number of functions that implement forward propagation and backpropagation in multi-layer perceptrons, as well as functions to train models using online learning and batch learning.

> module MLP_utils
>     ( evalEpoch
>     , evalEpochB
>     , batchSize
>     , feedFrwd
>     , toLoss
>     ) where

> import Control.Parallel -- for 'par' and 'pseq'
> import Data.List as L -- transpose to accumulate the error passed back to neurons in the previous layer
> import Types -- all custom types/classes for parallel multi-layer perceptrons
> import Data.List.Split as LS -- chunksOf function for batch learning in evalEpochB

The first function, nextStage, captures a common pattern used when feeding forward and back propagating through a neural network (see the definitions of feedFrwd and backProp).
The first argument, (l -> [NrnSignal] -> l'), is a function which takes an unprocessed layer l and signals (e.g. outputs or error) from the last layer that was processed, and returns the result of processing layer l.
The second argument, [l], represents the network at its previous stage of evaluation (a list of unprocessed layers). When nextStage is used to feed forward, the layers have not yet been activated so [l] :: [UnactivatedLayer]; when nextStage is used to back propagate, the layers have already been activated so [l] :: [ActivatedLayer].
The third argument, [NrnSignal], is the initial signal for the next stage of evaluation (either network inputs or prediction error). In the definition of the function, foldl is used to integrate list reversal with layer processing, effectively turning the accumulator argument into a stack so that state from the previous processed layer can be taken efficiently from the head.
Put simply, nextStage evaluates the network by processing it and reversing it so that it is ready for the next stage of evaluation:

> nextStage :: (Layer l, PropagatingLayer l') => (l -> [NrnSignal] -> l') -> [l] -> [NrnSignal] -> [l'] 
> nextStage f (x:xs) s = foldl' propagate [f x s] xs
>     where propagate zs@(y:ys) x = f x (signals y) : zs

By using nextStage it is easy to define the feed forward function and back propagation function. Each function need only pass the necessary arguments to the nextStage function: a function to process layers, the unprocessed network and a list of initial signals to propagate through the network:

> feedFrwd :: [UnactivatedLayer] -> [NrnSignal] -> [ActivatedLayer]
> feedFrwd uls inps = nextStage activLay uls inps
>    
> backProp :: [ActivatedLayer] -> [NrnSignal] -> [BackpropLayer]
> backProp als err = nextStage backpropLay als err 

Layers are activated using the activLay function, which takes an unactivated layer and previous layer outputs to produce outputs for the current layer and apply an activation function.

> activLay :: UnactivatedLayer -> [NrnSignal] -> ActivatedLayer
> activLay ul prev = activate . outputLay $ (ul,prev)

Outputs for the current layer are obtained by computing for each neuron: the dot product between its inputs (is) and the weights connecting it to the previous layer (wss).

> outputLay :: (UnactivatedLayer, [NrnSignal]) -> (UnactivatedLayer, [NrnSignal], [NrnSignal])
> outputLay (UL size wss af, is) = (UL size wss af, is, map dotProd wss)
>     where dotProd ws = sum $ zipWith (*) (1.0 : is) (map fst ws)

If the current layer uses an AD activation function then it exploits automatic differentiation by computing its derivative while evaluating the activation function. 
For each neuron, an AD activation function produces a tuple of the form (output, derivative): so these tuples must be unzipped and placed in the appropriate fields of an ActivatedLayer structure. 
If the current layer uses an ND activation function then a list of 1s is generated for the derivatives. This design choice was made since the output layer for MNIST models use softmax activation
with cross entropy loss, so we can use the toLoss function to compute the derivatives of cross entropy loss with respect to each [unactivated] output layer neuron (i.e. the dot product of its weights and inputs).
Computing derivatives this way is far easier than computing the derivatives for softmax outputs with respect to each [unactivated] output layer neuron in this function and then 
computing cross entropy loss with respect to softmax outputs in the toLoss function. Check the definition of toLoss to see why this is the case.

> activate :: (UnactivatedLayer, [NrnSignal], [NrnSignal]) -> ActivatedLayer
> activate (UL sz wss af, is, os) = case af of 
>                                     AD f -> let (os',ds) = unzip $ f os in AL sz wss af is os' ds         -- activation that includes AD - Automatic Differentiation
>                                     ND f -> let os'      = f os         in AL sz wss af is os' (repeat 1) -- activation that includes ND - No Differentiation
                      
The cross entropy loss function could be defined as: crossEntropy = [-(y * logBase 2 y') | (y,y') <- zip ys ys']
However, activate has been implemented so that we can use toLoss to compute derivatives of Cross Entropy Loss with respect to unactivated neuron outputs, giving us a
simpler definition:

> toLoss :: ([Double],[Double]) -> [Double]
> toLoss (ys,ys') = zipWith (-) ys ys'               

Error is propagated from one layer to the previous layer using the backpropLay function; this is done by propagating error from each neuron in the current layer to the previous layer using backNrn. 
To fully propagate error to the previous layer, each neuron in the previous layer must be given the sum of all errors propagated to it by neurons in the current layer; this is handled by accumError (in the where clause of backpropLay).
Layers are backpropagated in parallel by partitioning the set of neurons into 'nthreads' sets of neurons. Each thread works in parallel on its own allocated set(s) and when finished their results are concatenated together. 

> backpropLay :: ActivatedLayer -> [NrnSignal] -> BackpropLayer
> backpropLay (AL size wss af is _ ds) err = f `par` (e `pseq` (toBL . accumError . unzip $ e ++ f))
>     where 
>     f = forceList $ drop k input
>     e = forceList $ take k input 
>     k = ceiling $ fromIntegral size / fromIntegral nthreads
>     input = map (backNrn is) (zip3 ds err wss)
>     accumError (wss',ess) = (wss', tail . accum $ ess) -- tail ensures that error is not propagated to a bias neuron.
>     toBL (wss',es) = BL size wss' af es

Note that currently this module only supports 1 or 2 threads, so to use 1 thread and run this code sequentially set useParallelism to False, otherwise to use 2 threads and run this code with parallelism
set useParallelism to True.

> useParallelism = True
>
> nthreads = case useParallelism of
>               False -> 1
>               True  -> 2

To ensure that during parallelization a list is fully evaluated to Head Normal Form (rather than to WHNF - Weak Head Normal Form), we use a forceList function which "forces every element of a list to be evaluated to WHNF": http://book.realworldhaskell.org/read/concurrent-and-multicore-programming.html#id675076 
This is important because if skipped then another thread will need to evaluate the list to Head Normal Form which will unbalance the parallel workload distribution. Note that for each element x in xs: pseq will tell the compiler that x must be evaluated to WHNF: https://hackage.haskell.org/package/parallel-3.2.2.0/docs/Control-Parallel.html

> forceList :: [a] -> [a] 
> forceList []     = []
> forceList (x:xs) = x `pseq` (x : forceList xs)

The backNrn function is given the derivative, error, weights and inputs for a neuron n: the derivative and error of n is used to compute the error gradient, which is used (along with the weights and inputs of n) to update the weight of each connection between n and the previous layer. 
Each connection is updated by applying the weight and momentum update equation defined in appEquation.

> backNrn :: [Double] -> (Derivative,NrnSignal,NrnWeight) -> (NrnWeight,[NrnSignal])
> backNrn is (d,e,ws) = (zipWith (appEquation g) (1.0 : is) ws, map ((*g) . fst) ws) -- (1.0 : is) ensures that bias neuron weights are updated. 
>     where g = gClip $ d * e -- d * e gives us the gradient
> 
> appEquation :: Double -> Double -> (Double, Double) -> (Double, Double)
> appEquation g i (w,dw) =  (w+dw', dw')
>     where dw' = dw * getAlpha + (i * g * getEta)

Note that gClip implements gradient clipping to prevent gradient explosion, which can occur e.g. when using sigmoid activation to learn non-linear functions.
To remove gradient clipping, the function can simply be defined as 'gClip = id'

> gClip :: Double -> Double
> gClip = id                        -- no gradient clipping
> --gClip g | g > 0.5    = 0.5      -- 0.5 gradient clipping
> --        | g < (-0.5) = (-0.5)
> --        | otherwise  = g

The accum function adds elements from the same index of each sublist in a given list; each sublist represents a list of errors propagated from a neuron in layer n to its input neurons in layer n-1, so the error for a neuron in layer n-1 is the sum of errors propagated back to it from the neurons that it outputs to in layer n. This operation is equivalent to summing over the columns of a matrix:

> accum :: [[NrnSignal]] -> [NrnSignal]
> accum xs = [sum x | x <- L.transpose xs]   

Note that getEta and getAlpha define values for the learning rate (the magnitude of a weight update) and alpha (the impact of momentum on a weight update), respectively:

> getEta :: Double
> getEta = 0.04
> getAlpha :: Double
> getAlpha = 0.25

With forward passing and backpropagation implemented, it is now possible to combine these in a function that trains the model. The first and simplest approach to this is online learning, where back propagation occurs after every forward pass i.e. everytime a sample is processed. 
Note that a sample is represented as a tuple containing a model input, a model target and a boolean specifying if that sample is validation data or not (in which case it is training data).
Since a sample is just one element of a dataset, and the accuracy of a model is judged on how well it performs on a dataset, we inspect if the sample is validation data or training data and then
add the error to the appropriate error accumulator (tE or vE, which is the mean squared training error or validation error over the current epoch, respectively).
The function below, evalSample, implements online learning in accordance with these specifications:

> evalSample ::  ((Double, Double), [UnactivatedLayer]) -> (([Double], [Double]), Bool) -> ((Double, Double), [UnactivatedLayer])
> evalSample ((tE,vE),model) ((input,target),valid) = case valid of 
>                                                         False -> ((tE + sqrErr es, vE), model')
>                                                         True  -> ((tE, vE + sqrErr es), model)
>     where 
>     all_outs = feedFrwd model input
>     model' = map (\(BL size wss af _) -> UL size wss af) $ backProp all_outs es
>     pred = outputs . head $ all_outs -- extracts the model's prediction
>     es = toLoss (target, pred) 

Note that by taking the root mean squared error, all error values added to the accumulators are positive.

> sqrErr :: [Double] -> Double 
> sqrErr es = (sqrt . average) $ map (**2) es 
> 
> average :: Fractional a => [a] -> a 
> average [] = 0
> average xs = sm / ln 
>     where (sm, ln) = foldr (\x (y, i) -> (x+y, i+1)) (0,0) xs

The alternative approach to online learning is batch learning, in which backpropagation should occur after n forward passes (each on a different training sample) and the summed/mean error of all n predictions should be used for backpropagation.
The main difference to note here is that we have a list of data samples and validation booleans: so we need to run a forward pass on each data sample, partition the results into validation and training passes, sum up (and then normalize) the 
error for each set of passes, sum up (and then normalize) the model inputs and derivatives for the training data to use with the training error for backpropagation, return the updated validation and training errors.
For parallelization, a batch is partitioned into 2 sets which are forward propagated and aggregated in parallel, then their results are aggregated and normalized sequentially.

> evalBatch ::  ((Double, Double), [UnactivatedLayer]) -> ([([Double], [Double])], [Bool]) -> ((Double, Double), [UnactivatedLayer])
> evalBatch ((tE,vE),model) (trdata,valids) = f `par` (e `pseq` result)
>     where
>     f               = forceList . evalBatch' model $ drop k trdata
>     e               = forceList . evalBatch' model $ take k trdata
>     result          = case tBatch of 
>                          []        -> ((tE,       vE + vE'), model)
>                          otherwise -> ((tE + tE', vE + vE'), model')
>     (vBatch,tBatch) = separate valids $ e ++ f
>     --(aLays,tErrs)   = aggregateBatch tBatch                                    -- summed error
>     (aLays,tErrs)   = normalizeBatch (length tBatch) $ aggregateBatch tBatch   -- average error
>     k               = ceiling $ fromIntegral batchSize / fromIntegral nthreads
>     --tE'             = sqrErr tErrs                                             -- summed error
>     tE'             = sqrErr tErrs * fromIntegral batchSize                    -- average error
>     vE'             = sqrErr . snd $ aggregateBatch vBatch
>     model'          = map (\(BL size wss af _) -> UL size wss af) (backProp aLays tErrs)

To simplify the function above and increase modularity, we define a few helper functions below. The first, evalBatch', is used to perform a forward pass for every data sample in the batch.

> evalBatch' ::  [UnactivatedLayer] -> [([Double], [Double])] -> [([ActivatedLayer], [NrnSignal])]
> evalBatch' model trData = do
>     (input,target) <- trData
>     let activated = feedFrwd model input
>     let pred = outputs . head $ activated
>     let es = toLoss (target, pred) 
>     return (forceList activated, forceList es)

The next function, aggregateBatch, is used to sum up the model inputs and derivatives as well as the model error for each forward pass in a batch. Note that ActivatedLayer structures
are monoids which use mappend to accumulate inputs and derivatives for batch learning. The base cases of the aggregate function are used to catch cases where we try to aggregate a valid 
tuple with a tuple of empty lists (e.g. if batchSize is set to 1 when evaluating a batch then 'f' in the where clause of evalBatch will produce a tuple of empty lists).

> aggregateBatch :: [([ActivatedLayer],[NrnSignal])] -> ([ActivatedLayer],[NrnSignal])
> aggregateBatch []     = ([],[])
> aggregateBatch bs = foldl1' aggregate bs
>     where aggregate (lay,es) ([],[])    = (lay,                   es                )
>           aggregate ([],[])  (lay,es)   = (lay,                   es                )
>           aggregate (lay,es) (lay',es') = (zipWith (<>) lay lay', zipWith (+) es es')

The next function, normalizeBatch, is used to normalize an aggregated/summed batch by dividing model inputs, model derivatives and error by the number of samples that were in the batch i.e. n.

> normalizeBatch :: Int -> ([ActivatedLayer],[NrnSignal]) -> ([ActivatedLayer],[NrnSignal])
> normalizeBatch 0 (model,es) = (model                                                             , es  )
> normalizeBatch n (model,es) = ([AL sz wss af (f is) os (f ds) | (AL sz wss af is os ds) <- model], f es)
>     where f = map (/fromIntegral n)

Finally, we use the separate function to partition forward passes into validation and training passes, using a list of validation booleans to decide which set a pass is put into.

> separate :: [Bool] -> [a] -> ([a], [a])
> separate bs xs = foldl' f ([],[]) $ zip bs xs
>     where f (xs,xs') (True, x) = (x:xs, xs')
>           f (xs,xs') (False,x) = (xs, x:xs')

Now that we have implemented a function that evaluates a sample for online learning, and a function that evaluates a batch for batch learning, we need to implement functions that
fold these functions over a dataset (either a list of samples or a list of batches). Since the functions implemented above already add the error computed for a sample/batch to 
accumulated error arguments, we only need to initialize the errors to (0,0) and feed in the initialized model before folding over the dataset.

> evalEpoch :: [UnactivatedLayer] -> ([([Double], [Double])], [Bool]) -> ((Double, Double), [UnactivatedLayer])
> evalEpoch model (trData,valid) = foldl' evalSample ((0,0), model) $ zip trData valid     

> evalEpochB :: [UnactivatedLayer] -> ([([Double], [Double])], [Bool]) -> ((Double, Double), [UnactivatedLayer])
> evalEpochB model (trData,valid) = foldl' evalBatch ((0,0), model) $ zip (chunksOf batchSize trData) (chunksOf batchSize valid)

> batchSize :: Int
> batchSize = 20