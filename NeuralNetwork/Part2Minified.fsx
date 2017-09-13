#r @"../packages/MathNet.Numerics.3.20.0/lib/net40/MathNet.Numerics.dll"
#r @"../packages/MathNet.Numerics.FSharp.3.20.0/lib/net40/MathNet.Numerics.FSharp.dll"

open MathNet
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open System

let X = matrix [[0.; 0.; 1.]
                [0.; 1.; 1.] 
                [1.; 0.; 1.] 
                [1.; 1.; 1.;]]
let Y = (matrix [[0.; 1.; 1.; 0.]]).Transpose()
let weights0 = DenseMatrix.randomSeed 3 4 0
let weights1 = DenseMatrix.randomSeed 4 1 0
let rec train  (w0 : Matrix<float>) (w1 : Matrix<float>) (c : int) =
    let layer1 = 1./(1. + exp(-(X * w0)))
    let layer2 = 1./(1. + exp(-(layer1 * w1))) 
    let layer2delta = (Y - layer2) .* (layer2 .* (1. - layer2))
    let layer1delta = (layer2delta * w1.Transpose()) .* (layer1 .* (1. - layer1))
    if c = 60000 then layer2
    else train (w0 + X.Transpose() * layer1delta) (w1 + layer1.Transpose() * layer2delta) (c+1)
let Y' = train weights0 weights1 0