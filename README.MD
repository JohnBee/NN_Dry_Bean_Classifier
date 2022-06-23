# Dry Bean Dataset Classifier
A Classifier that operates on the Dry Bean Dataset from UCI Machine Learning Dataset Repository.

# Requirements
- Python > 3.6
- Numpy
- Pandas
- SciKit-Learn
- Matplotlib

# Dry Bean Dataset

[https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)

Reference:

KOKLU, M. and OZKAN, I.A., (2020), “Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques.” Computers and Electronics in Agriculture, 174, 105507.
DOI: https://doi.org/10.1016/j.compag.2020.105507

## Data cleanup
Exported the .xlsx Dataset as a .csv to be read in to the Python Script.

The data was checked for missing value and none wear found. This corroborates with the dataset description file.

A Scatter matrix was used to explore the data. There can be several features seen which strong correlation that perhaps could be removed.

<picture>
<img alt="scatter plot" src=https://github.com/JohnBee/NN_Dry_Bean_Classifier/blob/main/plots/correlation_scatter_plot.png width=800 />
</picture>

Scale the data using Min-Max Scaler and Apply PCA

PCA (Principal Component Analysis)

## Neural Network Parameters
Grid Search was used to find parameters for the MLP Classifier.
```commandline
Parameters:
Hidders Layer Architecture: (50, 50, 50)
activation: tanh
solver: adam
learning_rate: constant
alpha: 0.0001
```

The performance with PCA was lower than the performance without PCA applied.
```commandline
Accuracy Without PCA: 93%
Accuracy With PCA: 91%
```
Therefore PCA was removed.
## Performance
### Training set Confusion plot
The confusion plot shows a strong correct classification rate with many incorrect classifcations being small with exception to the incorrect prediction that a seed of type SIRA is a DERMASON.

<picture>
<img alt="train_confusion_plot_post_pca_removal" src=https://github.com/JohnBee/NN_Dry_Bean_Classifier/blob/main/plots/training_confusion_plot_1.png width="400"/>
</picture>

## Final test set performance
Accuracy Score for Test Set: 91% Accuracy which is the same to the accuracy recorded in the research paper when using MLP for Dry Bean Classification.
```commandline
Accuracy: 91.04%
```
### Final Test Set confusion plot:
Again, the MLP Classifer incorrectly classifies some SIRA as DERMASON, further work may be required to identify the cause.

<picture>
<img alt="test_confusion_plot" src=https://github.com/JohnBee/NN_Dry_Bean_Classifier/blob/main/plots/test_confusion_plot_1.png width=400/>
</picture>