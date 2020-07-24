# Music Featurization

This is the featurization step for the project "Walking through the Music and Visual Art StyleSpace via Computational Intelligence" (1).

Here we extracted the features from the penultimate layer of the network implemented by the author Hareesh Bahuleyan (2)(3), and we retrained the network usign the Magnatagatune dataset (4).

1. https://github.com/erikagardini/Walking-through-the-Music-and-Visual-Art-StyleSpace-via-Computational-Intelligence
2. Bahuleyan, H. Music Genre Classification using Machine Learning Techniques. arXiv:1804.01149
3. https://github.com/HareeshBahuleyan/music-genre-classification
4. http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
## Downloads files

```
git clone https://github.com/erikagardini/MusicFeaturization.git
```

## Install python requirements

You can install python requirements with

```
pip3 install -r requirements.txt
```

## Downloads the music dataset

From:

```
http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
```

Download the tag annotations (.csv file) and the audio data.
Drug the music directory and the tag annotations file inside the directory _datasets_ of this project:

<img src="https://github.com/erikagardini/MusicFeaturization/blob/master/extras/dir_example.png" width="200" />

## How to use the code

### Step 1: from mp3 to training, testing and validation data

```
cd datasets_utils
```

#### 1 - Get the spectrograms from the mp3 files

```
python3 1_generate_spectrograms.py
```

#### 2 - Assign genres and instruments to the spectrograms

```
python3 2_assign_genre.py
```

#### 3 - Select a subset of spectrograms

Select all the songs belonging to one of the following genres: 
- baroque 
- classical
- jazz
- medieval
- opera
- rock 

Establish a univocal class for each song by choosing the most specific (e.g. a song labelled as baroque and classical becomes baroque).

```
python3 3_remove_double_with_priority.py
```

#### 4 - Train, test, val split

- Training set -> 80% of the dataset
- Testing set -> 20% of the dataset
- Validation set -> 10% of the training set

```
python3 4_save_train_and_test.py
```

### Step 2: Train the model

__(This step can be skipped, the model of the trained networks is provided)__

```
cd ../python
python3 1-training.py
```

After each epoch, the weights of the current model are saved as "epoch_n_valCategoricalAccuracy.h5" (n is the current epoch number and _valCategoricalAccuracy_ is the current value of the categorical accuracy of the validation set) inside the directory _models_. 
When the training is completed, the file _history.pkl_ is saved inside the directory _models_ and contains the results of each epoch (training and validation loss/accuracy).

### Step 3: Choose the best model

Plot the loss/accuracy curves to simplify the choise of the best model:

```
cd ../results_utils
python3 plot_loss_accuracy_curves.py
```

![](https://github.com/erikagardini/MusicFeaturization/blob/master/results/loss.svg)
![](https://github.com/erikagardini/MusicFeaturization/blob/master/results/accuracy.svg)

In this case, the sixth 6 minimizes the validation loss and maximizes the validation accuracy.

### Step 4: Test the model

Test the model obtained after the sixth epoch.

```
cd ../python
python3 2-testing.py
```

When the testing is completed, the file _music_dataset_testing.csv_ is saved inside the directory _results_ and contains the output of the penultimate layer of the network during the testing.

### Step 5: Format the dataset

```
python3 3-format_dataset.py
```

When this step is completed, the file _music_dataset.csv_ is saved inside the directory _results_. This dataset is used for the experiment "Walking through the Music and Visual Art StyleSpace via Computational Intelligence" (1).

### Step 6: Analize the performance of the model

```
cd ../results_utils
python3 manage_prediction.py
```

When the step is performed, the file _res.txt_ is saved inside the directory _results_ which is a report containing the following metrics:

- TOP_1, TOP_2, TOP_3 Accuracy
- Matthews Coef
- F-score
- ROC AUC

Additionally, the confusion matrix is saved inside the directory _results_ (_conf_matr.png_).
