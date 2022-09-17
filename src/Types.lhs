> {-# LANGUAGE LambdaCase #-}

In this module are all of the custom types used by the deep learning library. Types are defined here to separate interface from implementation, this ensures that custom types are kept safe from modifications made to the modules which use them. 

> module Types
>     ( Layer
>     , PropagatingLayer
>     , weights
>     , size
>     , signals
>     , NrnWeight
>     , NrnSignal
>     , Derivative
>     , ActivationType (..)
>     , getActivationFunction
>     , ActivationFunction (..)
>     , sigmoid
>     , softmax
>     , UnactivatedLayer (..)
>     , ActivatedLayer (..)
>     , BackpropLayer (..)
>     ) where

> import Numeric.AD.Mode.Forward -- for grad'

> type NrnSignal = Double
> type NrnWeight = [(Double, Double)]
> type Derivative = Double

Each type of layer is defined as a custom data structure. In terms of maintaining the library this means that any changes resulting in errors within the modules that use these types are more likely to be caught at compilation time rather than manifesting as unexpected behaviour at runtime.
Each custom type has a size (which is used to parallelize backpropagation), weights and an activation function. In the training pipeline: UnactivatedLayers are converted into ActivatedLayers, which are converted into BackpropLayers, which are then converted back to UnactivatedLayers.

> data UnactivatedLayer = UL {ulSize :: Int, ulWeights :: [NrnWeight], ulActivation :: ActivationType} deriving (Show, Read)
> data ActivatedLayer   = AL {alSize :: Int, alWeights :: [NrnWeight], alActivation :: ActivationType, inputs :: [NrnSignal], outputs :: [NrnSignal], derivs :: [Derivative]} deriving (Show, Read)
> data BackpropLayer    = BL {blSize :: Int, blWeights :: [NrnWeight], blActivation :: ActivationType, errors :: [NrnSignal]} deriving (Show, Read)

Activation functions also have their own datatype. Since MNIST models use softmax activation for the output layer and cross entropy loss, the ND activation functions do not return derivatives - because 
it is much simpler to compute the derivatives of cross entropy with respect to unactivated output neurons in the toLoss function (defined in MLP_utils.lhs) rather than computing the derivatives of softmax 
with respect to unactivated output neurons here and using this to compute the derivatives of cross entropy with respect to unactivated output neurons. Sigmoid activation, which is used on the hidden layer 
of the MNIST model, is easily differentiated using the grad' function of the AD library, so this is implemented as an AD activation function. Note that due to the current structure of the library, only 
output layers can use ND activation, and can only use it if the toLoss function is responsible for implicitly computing derivatives. Read MLP_utils.lhs for more details.

> data ActivationType = Sigmoid | Softmax deriving (Show, Read)

> getActivationFunction :: ActivationType -> ActivationFunction
> getActivationFunction = \case
>   Sigmoid -> sigmoid
>   Softmax -> softmax

> data ActivationFunction = AD ([NrnSignal] -> [(NrnSignal, Derivative)]) | ND ([NrnSignal] -> [NrnSignal])

The sigmoid and softmax activation functions are defined below. As mentioned above, only the sigmoid activation function (which is used for hidden layers in the MNIST model) returns
derivatives. The AD library is used to exploit automatic differentiation by computing the derivatives for the sigmoid function whilst applying the sigmoid function to the outputs.

> sigmoid :: ActivationFunction
> sigmoid = AD $ \xs -> map (\x -> head <$> grad' sigf [x]) xs -- uses grad' to exploit automatic differentiation
>     where 
>     sigf [x] = 1 / (1 + exp 1 ** (-x))

> softmax :: ActivationFunction
> softmax = ND $ \xs -> [exp x / (sum . map exp $ xs) | x <- xs]

ActivatedLayers are Semigroups which are aggregated using the mappend operator (<>), as this simplifies the batch learning code in MLP_utils.lhs.

> instance Semigroup ActivatedLayer where
>     (AL sz wss af is os ds) <> (AL _ _ _ is' _ ds') = AL sz wss af (zipWith (+) is is') os (zipWith (+) ds ds')

Instances of the Layer class must contain a size, weights and activation functions. Instances of the PropagatingLayer class must also contain signals that can be propagated forwards or backwards through the network. During forward propagation, neurons take output signals from the previous layer. During backpropagation, neurons take error signals from the previous layer.

> class Layer a where
>     size       :: a -> Int
>     weights    :: a -> [NrnWeight]
>     activation :: a -> ActivationType
> class Layer a => PropagatingLayer a where
>     signals    :: a -> [NrnSignal]

> instance Layer UnactivatedLayer where
>     size       = ulSize
>     weights    = ulWeights
>     activation = ulActivation
> instance Layer ActivatedLayer where
>     size       = alSize
>     weights    = alWeights
>     activation = alActivation
> instance Layer BackpropLayer where
>     size       = blSize
>     weights    = blWeights
>     activation = blActivation

> instance PropagatingLayer ActivatedLayer where
>     signals = outputs
> instance PropagatingLayer BackpropLayer where
>     signals = errors
