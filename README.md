# Starbucks Capstone Project

## Project Motivation
This project was developed as part of the Udacity Data Scientist Nanodegree Capstone Project. Given a simulated dataset provided by Starbucks that mimics customer behavior on the Starbucks rewards mobile app, the task is to determine which demographic groups respond best to which offer type.

## Files Description
* `Starbucks_Capstone_notebook.ipynb` is the notebook for all the analysis and documentation of the developed solutions
* `app.py` contains the main code for running the web app
* `requirements.txt` contains list of dependencies for running the notebook and web app
* `docker-compose.yml` and `Dockerfile` are used to create the docker image to run the web app
* `utils` holds the utility functions used by the web app
    * `charts.py` contains code for creating the visualizations
    * `extract_transform.py` contains code for managing extraction and transformation tasks on the data
    * `inference.py` contains code for making the predictions
* `models` is the folder containing all fitted models used for inference
* `data` contains all the datasets used for the project (more details are provided in the notebook)
    * `portfolio.json`: containing offer ids and meta data about each offer (duration, type, etc.)
    * `profile.json`: demographic data for each customer
    * `transcript.json`: records for transactions, offers received, offers viewed, and offers completed


## Getting Started
### Running with Docker
To run the app, it suffices to have docker installed. Then, running the command bellow in the terminal inside the folder will build and start the docker image with the app. 
```
docker-compose up --build
```
If a web page doesn't automatically open up, type localhost:8501 in your browser. The app takes a few minutes to initialize.

Note: If you get the following message `starbucks-capstone_streamlit_1 exited with code 137`, try to increase the memory to around 8 GB.

### Running locally
You can also run the app locally without docker. For that, you'll need a basic installation of conda and the aditional packages listed in `requirements.txt` (plotly, xgboost, seaborn and streamlit).

After installing this libraries, you the running the following command in the terminal inside the folder will open the web app
```
streamlit run app.py 
```

## To Do List
- [ ] `nb` Refit model
- [ ] `nb` Create unit tests
- [ ] `nb` Assert effectiveness of sending offers (obs.: subject to confounder bias of sending the offer)
- [ ] `nb` Create feature of number of future offers sent and refit model