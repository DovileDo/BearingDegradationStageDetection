# A Framework for Predicting the Degradation Stages of Rolling-Element Bearings

For detailed description of the project please check https://dl.acm.org/doi/10.1145/3534678.3539057 (or preprint: https://arxiv.org/abs/2203.03259).

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
* train_NNclassifier.py train and save trained classifier (a separate classifier supervised by both AElabels and PCAlabels).

## Experiments
* train_NNclassifier.py to produce trainingacc.npg plot.
* AElabels_vs_Manual.py to produce Bearing1_1.png plot which compares single bearing AElabels to manual labels extracted manually examining changes in the frequency and RMS plots.
* LabelingPerformance.py to produce labelingacc.png which compares training dataset AElabels (Ytrain_AElabels.py) and PCAlabels (Ytrain_PCAlabels.py) to manual labels extracted manually examining changes in the frequency and RMS plots in Manual_labeling.py.
* ClassifierPerformance.py:
    * for bearing degradation stage posterior prediction in reports/figures/posterior
    * testAcc.png which compares classifier predictions to test set AElabels and PCAlabels.
    * overlap.png which shows the percentage of a degradation stage predicted by the classifier overlaped by any other bearing degradation stage.
    * fault_results.csv which shows how often the classifier predicts healthy bearing or degradation stage 1 after it predicted fault (stage 2 or 3) for the first time as well as bearing lifetime left after a fault was predicted.

### Built with

* Pandas 
* NumPy 
* SciPy
* Scikit-learn 
* Keras

## Citation
If you find our method useful in your research, please cite:

```yaml
@inproceedings{10.1145/3534678.3539057,
author = {Juodelyte, Dovile and Cheplygina, Veronika and Graversen, Therese and Bonnet, Philippe},
title = {Predicting Bearings Degradation Stages for Predictive Maintenance in the Pharmaceutical Industry},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539057},
doi = {10.1145/3534678.3539057},
abstract = {In the pharmaceutical industry, the maintenance of production machines must be audited by the regulator. In this context, the problem of predictive maintenance is not when to maintain a machine, but what parts to maintain at a given point in time. The focus shifts from the entire machine to its component parts and prediction becomes a classification problem. In this paper, we focus on rolling-elements bearings and we propose a framework for predicting their degradation stages automatically. Our main contribution is a k-means bearing lifetime segmentation method based on high-frequency bearing vibration signal embedded in a latent low-dimensional subspace using an AutoEncoder. Given high-frequency vibration data, our framework generates a labeled dataset that is used to train a supervised model for bearing degradation stage detection. Our experimental results, based on the publicly available FEMTO Bearing run-to-failure dataset, show that our framework is scalable and that it provides reliable and actionable predictions for a range of different bearings.},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {3107–3115},
numpages = {9},
keywords = {signal processing, predictive maintenance, dataset labeling, neural networks, data science, bearing degradation},
location = {Washington DC, USA},
series = {KDD '22}
}
```
