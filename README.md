# Overview

The aim of the project is to build a pipeline for experimenting and shipping results of an ML-powered web application. A simple [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) task is considered a good dummy to give focus to other parts of the project, namely:

- [MLflow](https://mlflow.org/): Automating tracking of experimentation phase, along with setting up a model registry to subsequently deploy the code;
- [Streamlit](https://streamlit.io/): Providing users with a web UI to interact with the code during the runtime;
- [Docker](https://www.docker.com/): Isolating model tracking and registering utilities with [MySQL](https://www.mysql.com/) being a backbone for storing metadata;
- [PyTorch](https://pytorch.org/): Making runtime faster and lighter with _distillation_ technique applied to initial classificator (multi-layer perceptron serving as a dummy, nothing fancy).

# Running the code

Before starting, one might want to create a new environment (`conda` is recommended) and install requirements with `conda install --file requirements.txt`.


## Tracking server

Firstly, MLflow tracking server and corresponding MySQL backend need to be up and running. One might want to adjust the `.env` file with the system-specific variables to alternate the default behaviour. Also, consider allowing writing permission to `MLFLOW_ARTIFACT_URI` directory to omit permission errors when storing model artifacts locally. 

Once ready, run:
```
docker-compose -f docker-compose.yaml -f docker-compose-local.yaml  up -d 
```
Please note that the interaction with the tracking server is done via __5005__ port by default. 


> For more details on how to set up the code for a remote server one may refer a great collection of MLflow use cases: https://github.com/amesar/mlflow-examples/tree/master/docker/docker-server.


## Experimenting over an ML task, using distilation for a lightweight model

Once both docker containers are up, proceed to the `/distillation/torch` folder and run `experimentation_torch.ipynb`. The notebook is configured to keep track of the experiments conducted in order to find the optimal parameters for the task at hand. Adding additional hyperparameters to the experimentation is straightforward and is left at one's discretion. It is also possible to overview the experimentation history at `http://localhost:5005/`.

As a result of the script, the best distilled model (as far as metics are conserned) is chosen for Staging (and, possibly, Productioning). The model is then served with `mlflow models serve -m "models:/{reg_model_name}/Staging" -p 5017 --no-conda &'` at `'http://127.0.0.1:5017/invocations'` MLflow tracking URI by default.

## Lighting the UI and testing the functionality

After the server is up with the selected model deployed, run `python streamlit/app.py' to launch the web application. A user is encouraged to load a Fashion MNIST image to the file uploading section in order to query an inference using the deployed model. An instance of the image may be found at `assets/Fashion_MNIST_images`.