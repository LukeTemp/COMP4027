> import MLP_utils -- for evalEpoch, evalEpochB and batchSize
> import InitModel -- for initWs, to randomly initialize models
> import Types -- all custom types/classes for parallel multi-layer perceptrons
> import Control.Monad.State.Lazy -- State Monad for implicit counting

The XOR function:

> xorData :: [([Double], [Double])]
> xorData = [([0.0,0.0],[0.0]), ([0.0,1.0],[1.0]), ([1.0,0.0],[1.0]), ([1.0,1.0],[0.0])]
>
> trainXORForN :: [UnactivatedLayer] -> State Int (Double, [UnactivatedLayer])
> trainXORForN model = do 
>     modify $ flip (-) 1
>     c <- get -- N of epochs remaining
>     --let ((tE,_),model') = evalEpochB model (xorData, map (\_ -> False) xorData)
>     let ((tE,_),model') = evalEpoch model (xorData, map (\_ -> False) xorData)
>     case c of 
>         0         -> return (tE,model')
>         otherwise -> trainXORForN model'
>
> trainXOR :: [UnactivatedLayer] -> IO ()
> trainXOR model = do
>     let (tE,model') = evalState (trainXORForN model) 20000 --100000
>     print $ "Training error from the latest Epoch: " ++ show tE
>     print $ "Model output for inputs [0.0,0.0]: " ++ (show . outputs . head $ feedFrwd model' [0.0,0.0])
>     print $ "Model output for inputs [0.0,1.0]: " ++ (show . outputs . head $ feedFrwd model' [0.0,1.0])
>     print $ "Model output for inputs [1.0,0.0]: " ++ (show . outputs . head $ feedFrwd model' [1.0,0.0])
>     print $ "Model output for inputs [1.0,1.0]: " ++ (show . outputs . head $ feedFrwd model' [1.0,1.0])

> randInitModel :: IO [UnactivatedLayer]
> randInitModel = initWs [2,2,1]

> preInitModel :: [UnactivatedLayer]
> preInitModel = [UL {ulSize = 2, ulWeights = [[(0.48383252051575276,0.0),(-0.3623918711857485,0.0),(-2.598485239650035,0.0)],[(-1.3294388861447821,0.0),(-1.2536278637555256,0.0),(-0.2728186257534778,0.0)]], ulActivation = sigmoid},UL {ulSize = 1, ulWeights = [[(0.1871333602666253,0.0),(-1.3452688344910684e-2,0.0),(-0.4517687835275232,0.0)]], ulActivation = sigmoid}]

We can either train a randomly initialized model or the pre-initialized model, uncomment the desired main function definition:

> main = randInitModel >>= trainXOR   -- train randomly initialized model
> --main = trainXOR preInitModel      -- train pre-initialized model

Configuration: getEta = 0.1, getAlpha = 0.25, gClip = id, use sqrErr in evalSample rather than sum, train for 20000 epochs