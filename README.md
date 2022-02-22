# A Framework for Predicting the Degradation Stages of Rolling-Element Bearings

As bearings degrade they go through physical changes that manifest in different frequency signatures in the frequency domain. Typically bearing degradation process is devided into five stages: 
* Healthy bearing, prominent fundamental frequencies;
* Degradation Stage 0, increase in ultrasonic frequencies (in this project considered to be healthy due to too high sampling frequency requirement);
* Degradation Stage 1, increase in natural frequencies;
* Degradation Stage 2, increase in fault freqencies.
* Degradation Stage 4, random noise type vibrations across the whole frequency spectra, especialy in the lower frequency range.

# Method

This bearing degradation stage detection method consists of two parts as shown in the chart below: 

* Domain knowledge based bearing vibration data labeling (Labeling step)
* Classifier trained with this labeled bearing vibration data for bearing degradation stage prediction (Unified classifier)

<img src="reports/figures/model.png" alt="method overview">

## Part 1 - Data labeling

The dataset used in the project need to be downloaded from https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#femto, and stored in the data directory.

Data preprocessing: 
* MergeDataFiles.py to merge files in the original data set to one file/bearing;
* TransformToFrequencyDomain.py to extraxt frequency domain features.

Data labeling (3 different labeling methods, Autoencoder-based labeling and PCA-based as well as manual labeling for reference):
* AutoEncoder.py for AutoEncoder-based data labeling.
* PCAlabeling.py for PCA-based data labeling.
* Ytrain_AElabels.py to extract AElabels for training data set.
* Ytrain_PCAlabels.py to extract PCAlabels for training data set.
* Manual_labeling.py to produce frequency and RMS plots for manual labeling.


## Part 2 - Bearing degradation stage classification

Feature extraction:
* TransformToFrequencyDomain.py and Xtrain_frequency.py to extraxt frequency domain features for classifier training.
* TransformToTimeDomain.py and Xtrain_time.py to extraxt time domain features for classifier training.

Model training:
* NNclassifier.py classifier architecture.
* train_NNclassifier.py train and save trained classifier.

## Experiments
* train_NNclassifier.py to produce trainingacc.npg plot.
* AElabels_vs_Manual.py to produce labelingacc.png plot which compares training dataset AElabels (Ytrain_AElabels.py) and PCAlabels (Ytrain_PCAlabels.py) to manual labels extracted manually examining changes in the frequency and RMS plots in Manual_labeling.py.
* ClassifierPerformance.py:
** for bearing degradation stage posterior prediction in reports/figures/posterior
** as well as testAcc.png 

### Built with

* Pandas 
* NumPy 
* SciPy
* Scikit-learn 
* Keras
