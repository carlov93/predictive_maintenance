# Manifold  
## Definition Manifold
- In non technical terms, a manifold is a continuous geometrical structure having finite dimension : a line, a curve, a plane, a surface, a sphere, ...
- A manifold is an object of dimensionality d that is embedded in some higher dimensional space. 
- Imagine a set of points on a sheet of paper. If we crinkle up the paper, the points are now in 3 dimensions. 
- Many manifold learning algorithms seek to "uncrinkle" the sheet of paper to put the data back into 2 dimensions.
- An other example is our planet Earth. For us it looks flat, but it reality it's a sphere. So it's sort of a 2d manifold embedded in the 3d space.
- A manifold is essentially a generalization of Euclidean space such that locally (small areas) are approximately the same as Euclidean space but the entire space fails to be have the same properties of Euclidean space when observed in its entirety.

## Definition of Dimensionality
- Dimensionality refers to the minimum number of coordinates needed to specify any point within a space or an object. 
- So a line has a dimensionality of 1 because only one coordinate is needed to specify a point on it. 
- A planar surface, on the other hand, has a dimensionality of 2 because two coordinates are needed to specify a point on it. So trying to locate ‘5’ on a surface is meaningless because you need to specify the other coordinate too.

## Manifold Hypothesis
- Although the data points may consist of thousands of features, they may be described as a function of only a few underlying parameters. 
- That is, the data points are actually samples from a low-dimensional manifold that is embedded in a high-dimensional space.
- The Manifold Hypothesis states that real-world high-dimensional data lie on low-dimensional manifolds embedded within the high-dimensional space.
- It explains (heuristically) why machine learning techniques are able to find useful features and produce accurate predictions from datasets that have a potentially large number of dimensions ( variables).
- The fact that the actual data set of interest actually lives in a space of low dimension, means that a given machine learning model only needs to learn to focus on a few key features of the dataset.
- However these key features may turn out to be complicated functions of the original variables.  Many of the algorithms behind machine learning techniques focus on ways to determine these (embedding) functions.


__Example:__
- Imagine we are interested in classify images with m x n pixels. 
- Each pixel has a numerical value, and each can vary depending on what the image is. 
- The point is that we have mxn degrees of freedom so we can treat an image of mxn pixels as being a single point in living in a space (manifold)  of dimension N = m x n,  
- Now, imagine the set of all m x n imagines that are photos of Einstein. 
- Clearly we now have some restriction on the choice of values for the pixels if we want the images to be photos of Einstein rather than something else. 
- Obviously random choices will not generate such images. Therefore, we expect there to be less freedom of choice and hence:
--> The manifold hypothesis states that that this subset should actually live in an (ambient) space of lower dimension, in fact a dimension much, much smaller than N. 


## Dimension reduction with non-linear manifold learner
- When dimensionality is very large (larger than the number of the samples in the dataset), we could run into some serious problems. 
- if we have more parameters than there are samples in the entire dataset, it means that a learner will be able to overfit the data, and consequently won't generalize well to other samples unseen during training.
- A non-linear manifold learner can produce a space that makes classification and regression problems easier.
