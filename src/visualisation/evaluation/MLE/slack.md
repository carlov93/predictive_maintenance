Hey, finally I can contribute something to the discussion. 
I write a master thesis about different novelty detection metrics for LSTM. The results below are from a LSTM model, trained with the log-likelihood loss function (enables us to predict mean and variance for each point). The ground truth in the phase with no error is 0 and 1 in the phase with small or large error. Values that lie outside of the predicted mean and +- 2 predicted sigma are classified as anomalies by the LSTM model. 
I used the data from *samples_obs_space_error_sinusiod* in *data_obsdim10_walkbias100_cccurviness1*.
This graph shows for each sensor the f1 score depending on delta t. Starting point is the beginning of the large error and from that point the f1 score is calculated for different step length. Prof. Niggemann's idea was to check after which elapsed time the model is convinced that the current time series should be defined as an anomaly. I made two observations: __1.__ Precision Score does not make sense in this setting, because no FP is possible in the phase with an error (ground truth is always 1). __2.__ I wonder about the constant f1 score. With a linearly increasing error (like it is implemented in the cpps_gendata), I would have assumed that the F1 score improves over time, as the model classifies more points as anomaly and thus approaches ground truth. But that is not the case. Is the slope too small? And I noticed, that the variance increases with the time as well. Is that what we want?
```{python}
errors = np.zeros(dim_latent_space * num_samples)
for i in range(num_samples):
    if i > error_latent_space_large_pos:
        error = np.random.normal(error_latent_space_large[0] * i,
                                 error_latent_space_large[1] * i,
                                 dim_latent_space)
samples_latent_space_error = samples_latent_space + errors
```