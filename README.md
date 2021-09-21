# starbucks-capstone
Udacity Data Scientist Capstone Project

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