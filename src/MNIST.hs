{-|
Module      : MNIST
Description : Functions for working with the MNIST dataset
License     : MIT
Maintainer  : andor.willared@mni.thm.de
Stability   : experimental

Provides functions for parsing <http://yann.lecun.com/exdb/mnist MNIST> data, aswell as additional functions for working with PNGSs.
-}


module MNIST (
    -- * Parsing
    getTrainingSamples,
    getTestSamples,
    getTrainingSamplesWithPaths,
    getTestSamplesWithPaths,
    pngToVector,
    -- * Rendering
    vectorToPNG,
    render
    ) where

import System.IO
import qualified Data.ByteString as B
import Data.Matrix
import Data.List.Split

import Codec.Picture.Types
import Codec.Picture.RGBA8
import Codec.Picture.Png


-- Parsing

-- | 'getTrainingSamples' gets mnist training samples if the needed files are in "./mnist/"
getTrainingSamples :: IO ([(Matrix Float, Matrix Float)]) -- ^ Training data as a list of pairs, where fst represents an image and snd the corresponding label

getTrainingSamples = getTrainingSamplesWithPaths "mnist/train-images.idx3-ubyte" "mnist/train-labels.idx1-ubyte"

-- | 'getTrainingSamples' gets mnist test samples if the needed files are in "./mnist/"
getTestSamples :: IO ([(Matrix Float, Matrix Float)]) -- ^ Test data as a list of pairs, where fst represents an image and snd the corresponding label

getTestSamples = getTestSamplesWithPaths "mnist/t10k-images.idx3-ubyte" "mnist/t10k-labels.idx1-ubyte"

-- | 'getTrainingSamplesWithPaths' is used to parse the raw MNIST training data to a representation usable in Haskell
getTrainingSamplesWithPaths :: FilePath                              -- ^ Path to "train-images.idx3-ubyte"
                            -> FilePath                              -- ^ Path to "train-labels.idx3-ubyte"
                            -> IO ([(Matrix Float, Matrix Float)])   -- ^ Training data as a list of pairs, where fst represents an image and snd the corresponding label

getTrainingSamplesWithPaths pathImgs pathLabels = do
  images <- parseImages pathImgs
  labels <- parseLabels pathLabels
  return (zip images labels)

-- | 'getTestSamplesWithPaths' is used to parse the raw MNIST test data to a representation usable in Haskel
getTestSamplesWithPaths :: FilePath                              -- ^ Path to "t10k-images.idx3-ubyte"
                        -> FilePath                              -- ^ Path to "t10k-labels.idx3-ubyte"
                        -> IO ([(Matrix Float, Matrix Float)])   -- ^ Test data as a list of pairs, where fst represents an image and snd the corresponding label

getTestSamplesWithPaths pathImgs pathLabels = do
  images <- parseImages pathImgs
  labels <- parseLabels pathLabels
  return (zip images labels)

-- | 'parseLabels' takes a FilePath and returns a list of matrixes representing the labels
parseLabels :: FilePath     -- ^ Path to a MNIST label file
            -> IO ([Matrix Float])
parseLabels path = do
  labels <- B.readFile path
  return (map (fromList 10 1) (map toCategorical10 (map fromIntegral (B.unpack (B.drop 8 labels)))))

-- | 'parseImages' works exactly like 'parseLabels' but returns a representation for images
parseImages :: FilePath
            -> IO ([Matrix Float])  -- ^ Path to a MNIST image file
parseImages path = do
  images <- B.readFile path
  return (map (fmap (/255)) (map (fromList 784 1) (chunksOf 784 (map fromIntegral (B.unpack (B.drop 16 images))))))


-- Png

-- |  'pngToVector' takes a file path and parses it to an equivalent float matrix
pngToVector :: FilePath             -- ^ Path to a .png file
            -> IO (Matrix Float)    -- ^ Float matrix representing the input file

pngToVector path = do
  pngData <- B.readFile path
  let decodedPng = decodePng pngData
  case decodedPng of
    Left err -> error (show err)
    Right succ -> return (fmap (/255.0) (fromList 784 1 (map fromIntegral ([redChannelAt (fromDynamicImage succ) x y | y <- [0..27], x <- [0..27]]))))

-- | 'vectorToPng' takes a float matrix and a file path, creates an image representation of the input matrix
vectorToPNG :: Matrix Float -- ^ Float matrix to write
            -> FilePath     -- ^ Path to write the .png to
            -> IO()

vectorToPNG vector path = writePng path (generateImage (grayscaleAt vector) 28 28)

-- | 'grayscaleAt' reads a value specified by X and Y from a matrix, returning a PixelRBA8 of the same brightness
grayscaleAt :: Matrix Float     -- ^ Matrix to read from
            -> Int              -- ^ X value
            -> Int              -- ^ Y value
            -> PixelRGBA8       -- ^ Grayscale RGBA8

grayscaleAt vector x y = PixelRGBA8 grayscale grayscale grayscale 255
  where grayscale = round ((getElem (x+y*28+1) 1 vector)*255)

-- | 'render' renders the input of a samples in the console
render :: Matrix Float  -- ^ Matrix to render
       -> IO ()

render matrix = do
  putStrLn (['\n'] ++ (insert '\n' 28 (map (\x -> if x /= 0 then '.' else ' ') (toList matrix))))

-- Helper

toCategorical10 :: Int -> [Float]
toCategorical10 label = [if i == label then 1 else 0 | i <- [0..9]]

-- | 'redChannelAt' reads the R value of a Pixel specified by X and Y from a given image
redChannelAt :: Image PixelRGBA8    -- ^ Image to read from
             -> Int     -- ^ X coordinate of desired pixel
             -> Int     -- ^ Y coordinate of desired pixel
             -> Int     -- ^ Red Channel of the RGBA8 at the desired coordinate

redChannelAt rgba8 x y = redChannel (pixelAt rgba8 x y)

-- | 'redChannel' returns R value of a PixelRGBA8
redChannel :: PixelRGBA8    -- ^ Pixel to read from
           -> Int           -- ^ Red Channel of the RGBA8

redChannel (PixelRGBA8 r g b a) = fromIntegral r

-- | 'insert' inserts a character e in string x at every nth position
insert :: Char      -- ^ Character to insert
       -> Int       -- ^ Frequency
       -> [Char]    -- ^ List to insert
       -> [Char]    -- ^ List after insertion
insert _ _ [] = []
insert e n x = ((take n x) ++ [e]) ++ (insert e n (drop n x))
