## Next steps on programming site
1. Understand the difference between prediction on new blade data and worn blade data

2. Check sensor data regarding random walk (check each sensor time series by looking at the autocorrelation)
   - if there is no autocorrelation between the difference x_t and x_t-1, it is a perfect random walk
   - if the autocorrelation of the original time series is -1, it is a perfect random walk (model uses last value to predict next value)