# Types of time series forecasting problems
## Univariate time series forecasting
These are problems comprised of a single series of observations and a model is required to learn from the series of past observations to predict the next value in the sequence. Thus, in univariate series, number of features is one, for one variable.

An univariate sequence looks like this: <br>
`[10, 20, 30, 40, 50, 60, 70, 80, 90]`

## Multivariate time series forecasting
Multivariate time series data means data where there is more than one observation for each time step. 
A problem may have two or more parallel input time series and an output time series that is dependent on the input time series.
The input time series are parallel because each series has an observation at the same time steps.
This is the case if 3 sensors measure at the same time, each produces a time series and for every time step you get 3 observations. 

|t |X |y|
|--|--|---|
t_1|10, 20, 30	|	40
t_2|20, 30, 40	|	50
t_3|30, 40, 50	|	60

Sensor 1: 10, 20, 30 <br>
Sensor 2: 20, 30, 40 <br>
Sensor 3: 30, 40, 50 <br>

## Multi-step (multivariate) forecasting
A time series forecasting problem that requires a prediction of multiple time steps into the future can be referred to as multi-step time series forecasting.

Specifically, these are problems where the forecast horizon or interval is more than one time step.
