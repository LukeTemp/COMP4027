In this module are a number of functions used to implement random dataset shuffling for Stochastic Gradient Descent. 

> module SGDF
>     ( shuffle
>     ) where

> import Control.Monad.State.Lazy -- for the State Monad
> import System.Random -- for mkStdGen
> import GHC.Word -- for genWord32R

The shuffle function below is used to randomly shuffle datasets for SGD. The helper function shuffle' uses a counter to track the remaining length of the 'unshuffled list', and uses 
this counter to randomly select indices (obtained using genWord32R) at which elements will be popped from the unshuffled list and appended onto the new list i.e. the shuffled list.

> shuffle :: [a] -> [a]
> shuffle xs = evalState (shuffle' (mkStdGen 0) xs) (fromIntegral $ length xs - 1) 

> shuffle' :: RandomGen g => g -> [a] -> State Word32 [a] 
> shuffle' _ [] = return []
> shuffle' g xs = do
>     n <- get 
>     let (v,g') = genWord32R n g
>     let (y,xs') = popN (fromIntegral v) xs
>     put (n-1)
>     ys <- shuffle' g' xs'
>     return $ y : ys 

> popN :: Int -> [a] -> (a,[a])
> popN n xs = runState (popN' n xs) []
    
> popN' :: Int -> [a] -> State [a] a 
> popN' 0 (x:xs) = do 
>     put xs 
>     return x  
> popN' n (x:xs) = do
>     x' <- popN' (n-1) xs
>     xs' <- get
>     put (x:xs')
>     return x'   

