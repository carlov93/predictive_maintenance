# Data preperation for time series problems
## Construct sliding window
- The LSTM model will learn a function that maps a sequence of past observations as input to an output observation. As such, the sequence of observations must be transformed into multiple examples from which the LSTM can learn.
- Samples are constructed using a sliding window with step size one, where each sliding window contains the previous n days as input, and aims to forecast the upcoming day. 

__Example with an univariate sequence:__ <br>
An univariate sequence looks like this: <br>
```[10, 20, 30, 40, 50, 60, 70, 80, 90]```

We can divide the sequence into multiple input/output patterns called samples, where three time steps are used as input and one time step is used as output for the one-step prediction that is being learned.

X |y
--|---
10 20, 30	|	40
20, 30, 40	|	50
30, 40, 50	|	60

## Log-transformation
- The raw data are log-transformed to alleviate exponential effects. 

## Remove trends
- Next, within each sliding window, the first day is subtracted from all values, so that trends are removed and the neural network is trained for the incremental value. 

## Test Time
- At test time, it is straightforward to revert these transformations to obtain predictions at the original scale.

## Source 
Paper (Deep and Confident Prediction for Time Series at Uber)