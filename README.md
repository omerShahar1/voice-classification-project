## Introduction
The following project focus on age and gender classification from voice. 
We created two models with the objective to detect the age and gender of the users through chosen traits in their voice. 
* The ages groups we used for classification are: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
* the gender classification groups are male and female.


## Dataset used

[Mozilla's Common Voice](https://www.kaggle.com/mozillaorg/common-voice) 

**Note that the following is optional**
If you wish to extract the features (the npy files) on your own: dwanload the dataset, extract it and move it with the files currently in "archive" folder. Then you will need:
1. run mp3_to_wav.py
2. run  preparation.py
3. run save_csv.py


## Training
* In order to train the age classification model, make sure the "is_age" flag is True (line 11 "train.py"). 
* If you want to train the gender classification model, make sure the same flag is False.
To train a model run the following command:

    python train.py

## Testing

[`test.py`](test.py) is the code responsible for testing your audio files or your voice:

    python test.py



## results accuracy
* gender classification success rate is currently 96%
* age classification success rate is currently 78%


