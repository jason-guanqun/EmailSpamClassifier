# Some projects of machine learning course
## EmailClassifier
An email classier to detect spam emails
### Instructions
The data file contains two folders: train_data and test_data.
There are a total of about 4,400 emails for training in train_data folder placed in two subfolders ham and spam. Emails in the ham folder are considered legitimate or not-spam, while the emails in the spam folder are considered spam. Each email is a separate text file in these subfolders. These emails have been slightly preprocessed to remove meta-data information.
## MusicClassifier
### Description
The given dataset comes from musical songs domain. It consists of about 1/2 Million songs from 20th and 21st century. The task is to predict the year the song was released given the musical content of the song.

To ease the preprocessing burden, each song (observation) has already been vectorized into 90 high quality timbre based features.
### Instructions
The data file contains three variables: trainx, trainy, testx.

- trainx is a 463715×90 matrix of training data, where each row is a 90-dimensional feature representation of a song.
- trainy is a 463715×1 vector of labels associated with each training data. Each row is the release year of the corresponding song in trainx.
- testx is a 51630×90 matrix of test data.
