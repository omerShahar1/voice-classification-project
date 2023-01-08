
## Dataset used

[Mozilla's Common Voice](https://www.kaggle.com/mozillaorg/common-voice) 

If you wish to extract the features (the npy files) on your own: dwanload the dataset, extract it and move it with the files currently in "archive" folder. Then you will need:
1. run mp3_to_wav.py
2. run  preparation.py
3. run save_csv.py

## Training
If you want to train the age_classification model, make sure that "is_age" flag is True (line 11 "train.py"). If you want to train the gender_classification model, make sure the flag is False.
To train a model run the following command:

    python train.py

## Testing

[`test.py`](test.py) is the code responsible for testing your audio files or your voice:

    python test.py

    
