



Paper (Deep and Confident Prediction for Time Series at Uber)
- Samples are constructed using a sliding window with step size one, where each sliding window contains the previ- ous 28 days as input, and aims to forecast the upcoming day. 
- The raw data are log-transformed to alleviate exponential effects. 
- Next, within each sliding window, the first day is subtracted from all values, so that trends are removed and the neural network is trained for the incremental value. 
- At test time, it is straightforward to revert these transformations to obtain predictions at the original scale.