# Time Series
## Why time series data is unique
- A time series is a series of data points indexed in time. 
- The key point about time series data is that the ordering of the time points matters. For many sets of data (for example, the heights of a set of school children) it does not really matter which order the data are obtained or listed. One order is good as another. For time series data, the ordering is crucial.
- Serial dependence occurs when the value of a datapoint at one time is statistically dependent on another datapoint in another time. 
- However, this attribute of time series data violates one of the fundamental assumptions of many statistical analyses — that data is statistically independent.
   - In probability theory, two events are statistically independent if the occurrence of one does not affect the probability of occurrence of the other 

## Time Series Components
__Trend:__ <br>
- A trend exists when a series increases or decreases with respect to time. Therefore, the time is taken as a feature. 

__Seasonality:__ <br>
- This refers to the property of a time series that displays periodical patterns that repeats at a constant frequency (m)

## Residuals diagnostics
- After a forecasting model has been fit, it is important to assess how well it is able to capture patterns. 
- While evaluation metrics help determine how close the fitted values are to the actual ones, they do not evaluate whether the model properly fits the time series.
- People often use the residuals to assess whether their model is a good fit while ignoring that assumption that the residuals have no autocorrelation
- As you are trying to capture the patterns of a time series, you would expect the errors to behave as white noise, as they represent what cannot be captured by the model. 
- White noise must have the following properties:
   - The residuals are uncorrelated (Acf = 0)
   - The residuals follow a normal distribution, with zero mean (unbiased) and constant variance
   
- the first property can be verified like this:
   - Plot the Autocorrelation function (ACF) and evaluate that at least 95% of the spikes are on the interval $-\frac {2}{\sqrt{T}}, \frac {2}{\sqrt{T}}$ where T is the size of the time series.
- If either of the two properties are not present, it means that there is room for improvement in the model.

![](../pictures/interval.png)

## Autocorrelation
- Autocorrelation is a measure of the internal correlation within a time series
- It is a type of serial dependence. Specifically, autocorrelation is when a time series is linearly related to a lagged version of itself. 
- Values closer to plus or minus one indicate strong correlation
- For example, for the time series you might ask: “take an arbitrary spot in time, on the average what does the time series look like in four weeks time, compared to now? ”
- A negative autocorrelation implies that if a particular value is above average the next value (or for the number of the lag) is more likely to be below average. 

## Stationary Time Series
- A stationary time series is one whose statistical properties are constant over time:
   - the mean value of the series
   - the variance
   - and the autocorrelation. 






