> module Types 
>     ( Layer
>     , PropagatingLayer
>     , weights
>     , signals
>     , NrnWeight
>     , NrnSignal
>     , UnactivatedLayer (..)
>     , ActivatedLayer (..)
>     , BackpropLayer (..)
>     ) where

Types are defined in this module to separate interface from implementation, this ensures that custom types are kept safe from modifications made to the modules which use them. 

> type NrnSignal                      = Double
> type NrnWeight                    = [(Double, Double)]

Each type of layer is defined as a custom data structure, so that any problems in the modules that use these types are more likely to be caught at compilation time rather than manifesting as unexpected behaviour at runtime.

> newtype UnactivatedLayer = UL {weights1 :: [NrnWeight]}
> data ActivatedLayer             = AL {weights2 :: [NrnWeight], inputs :: [NrnSignal], outputs :: [NrnSignal]}
> data BackpropLayer            = BL {weights3 :: [NrnWeight], curErr :: [NrnSignal]}

Instances of the Layer class must contain weights.

> class Layer a where
>     weights :: a -> [NrnWeight]
> class Layer a => PropagatingLayer a where
>     signals :: a -> [NrnSignal]
    
> instance Layer UnactivatedLayer where
>     weights = weights1
> instance Layer ActivatedLayer where
>     weights = weights2
> instance Layer BackpropLayer where
>     weights = weights3

Instances of the PropagatingLayer class must contain signals that can be propagated forwards or backwards through the network. During forward propagation, neurons take output signals from the previous layer. During backpropagation, neurons take error signals from the next layer.
    
> instance PropagatingLayer ActivatedLayer where
>     signals = outputs
> instance PropagatingLayer BackpropLayer where
>     signals = curErr

Layers should be printed in a common format:

> instance Show UnactivatedLayer where
>     show (UL ws) = "UL {weights = " ++ show ws ++ "}"
> instance Show ActivatedLayer where
>     show (AL ws is os) = "AL {weights = " ++ show ws ++ ", inputs = " ++ show is ++ ", outputs = " ++ show os ++ "}" 
> instance Show BackpropLayer where
>     show (BL ws es) = "BL {weights = " ++ show ws ++ ", errors = " ++ show es ++ "}"