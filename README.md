## Purpose of the project and team members 
This project aims to generate music with Chopin style. 
Contributors to the project are :
* Sina BEHDADNIA 'https://github.com/sinabehdadnia'
* Anh Dao HO 'https://github.com/ChiiDao'
* Zoé KOLAN 'https://github.com/zoekolan'
* Aurélie RAYMOND 'https://github.com/aeraymd'

## Structure of the project 
Three models were tested during the projects, but only two gave satisfactory results.
* LSTM model with attention layers ==> LSTM_model_with_attention_layers.ipynb
* GPT (most successful results) ==> GPT_model.ipynb

Both workflows are integrated in the notebooks. 
Beware that the training part can take a long time (30 mn to several hours depending on computation power). 
Paths for inputs and outputs need to be updated accordingly. 

Other directories include : 
* package for preprocessing and training part of the GPT model
* subfolders for exploratory work
* function to convert MIDI files with two channels (left hand/right hand in the case of piano music) into one channel 

[Watch the presentation on youtube](https://www.youtube.com/watch?v=3Pepuk1x-Jw)
