# zip-mnist 

Repository to test different compression algorithms on MNIST and their effect on downstream performance. 
Algorithms reduce the number of features by half (from 784 to 400). 

# Results 

| name            |   train_accuracy |   test_accuracy |   total_training_time |   total_time |
|:----------------|-----------------:|----------------:|----------------------:|-------------:|
| baseline        |         0.963317 |          0.9422 |               5.16806 |      5.46525 |
| linear_sparsity |         0.9596   |          0.9416 |               1.15002 |      1.36191 |
| zip             |         0.5998   |          0.5514 |               1.92852 |      7.38709 |
| pca             |         0.941217 |          0.9132 |               1.83871 |      2.29162 |
| autoencoder     |         0.932    |          0.9034 |               1.6643  |     37.5491  |


# Explanations: 

- baseline: just passes the features to xgboost. 
- linear_sparsity: skips features with an np.linspace function 
- zip: uses zlib to create a binary buffer and converts it back into a numpy array 
- pca: standard pca from sklearn 
- autoencoder: tensorflow with dense layers trained with keras. 

# Conclusions: 

- The simplest solution wins -- linear_sparsity reduces train time by 80% with a small loss in accuracy.
- zip compresses the data in a way that makes it hard to extract patterns -- big loss of accuracy and also slow 
- pca is decent but worse than linear_sparsity in every way
- autoencoder is even worse than pca and 10 times slower. 