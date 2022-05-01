# CS5425 Project: Russia-Ukraine Twitter Analytics

## Installation
Pliease install python modules based on the versions in requirements.txt.

- advice: if you don't wanna make your current conda enviroment messy, please create a new one as below:
    > conda create -n YOUR_ENV_NAME python=3.8
    >
    > conda activate YOUR_ENV_NAME

then, install all dependacies.
> pip install -r requirements.txt

## Run demo Locally
Note: as the data is not real-time(otherwise scraping is time-consuming), we **only give several sample *dates* of tweets data**.

> cd ProcessedData\&UI && streamlit run UI.py
