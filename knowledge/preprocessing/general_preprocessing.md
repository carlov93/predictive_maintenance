# Preprocessing
Good practice is as follows:
1. Fit the scaler using available training data. For normalization, this means the training data will be used to estimate the minimum and maximum observable values. This is done by calling the fit() function,
2. Apply the scale to training data. This means you can use the normalized data to train your model. This is done by calling the transform() function
3. Apply the scale to data going forward. This means you can prepare new data in the future on which you want to make predictions.

If needed, the transform can be inverted. This is useful for converting predictions back into their original scale for reporting or plotting. This can be done by calling the inverse_transform() function.