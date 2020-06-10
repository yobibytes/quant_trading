# install packages
```sh
sudo pip install -U kedro kedro-viz kedro-airflow kedro-docker
conda create --name py37 python=3.7
conda config --add channels conda-forge
sudo mount -t drvfs d: /mnt/d
cd /mnt/d/notebooks/bootcamps/kedro
kedro info
```
# initialize project

* kedro workflow: https://kedro.readthedocs.io/en/stable/03_tutorial/01_workflow.html
```sh
CONDA_ENV="py37"
conda activate $CONDA_ENV
sudo kedro new
cd hello_world
mv ./conf/base/credentials.yml ./conf/local/
```
```.env
CONDA_ENV="py37"
```
```.gitignore
# ignore all local configuration
conf/local/**
!conf/local/.gitkeep
 
# ignore potentially sensitive credentials files
conf/**/*credentials*
 
# ignore everything in the following folders
data/**
logs/**
references/**
results/**
```
# after init
```
# pip install -U -r src/requirements.txt
# conda install --file src/requirements.txt
# conda env export --from-history >src/environment.yml

source .env
conda activate $CONDA_ENV
sudo kedro install
kedro jupyter lab --NotebookApp.token='' --NotebookApp.password=''
kedro jupyter notebook
kedro lint
kedro test
# kedro black
# kedro lint
# kedro pre-push -blt
kedro run
kedro build-docs
kedro package
```
# test datasets
```
kedro ipython
context.catalog.load("shuttles").head()
exit()
```
# test nodes / pipelines
```
kedro run --node=preprocessing_companies
kedro run --pipeline=de
kedro run --env=test
# or: export KEDRO_ENV=test
#https://kedro.readthedocs.io/en/stable/04_user_guide/03_configuration.html#configuring-kedro-run-arguments
kedro run --config config.yml
kedro run
kedro run --parallel
```
# docker
```
# https://github.com/quantumblacklabs/kedro-docker
kedro docker build
# docker run -v ~/my-project:/sources <my-image>
kedro docker run
kedro docker run --docker-args="--env KEY=MYVALUE" --parallel
kedro docker jupyter lab --docker-args "-v ${PWD}:/home/kedro" --NotebookApp.token='' --NotebookApp.password=''
kedro docker jupyter notebook
kedro docker dive
# kedro viz
kedro docker cmd --docker-args="-p=4141:4141" kedro viz --host=0.0.0.0
# or ...
pip download -d data --no-deps kedro-viz
kedro docker build
kedro docker cmd bash --docker-args="-it -u=0 -p=4141:4141"
pip install data/*.whl
kedro viz --host=0.0.0.0 --no-browser
```
# airflow
```
# https://airflow.apache.org/docs/stable/installation.html
export AIRFLOW_HOME=~/airflow
sudo pip install -U 'apache-airflow[all]'
airflow initdb
airflow webserver -p 8080
airflow scheduler

# https://github.com/quantumblacklabs/kedro-airflow#prerequisites
kedro airflow create
# check airflow_dags folder: https://github.com/quantumblacklabs/kedro-airflow/blob/master/README.md#customization
kedro airflow deploy
```

# directory structure

├── .ipython/
├── conf/
├── data/
├── docs/
├── logs/
├── notebooks/
├── src
│   ├── new_kedro_project
│   │   ├── pipelines
│   │   │   ├── data_engineering: A pipeline that imputes missing data and discovers outlier data points
│   │   │   │   ├── __init__.py
│   │   │   │   ├── nodes.py: To ensure portability, modular pipelines should use relative imports when accessing their own objects and absolute imports otherwise.
│   │   │   │   ├── pipeline.py
│   │   │   │   ├── requirements.txt: A modular pipeline may have external dependencies specified in requirements.txt. These dependencies are not currently installed by the kedro install command
│   │   │   │   └── README.md: with all the information regarding the execution of the pipeline for the end users
│   │   │   ├── feature_engineering: A pipeline that generates temporal features while aggregating data and performs a train/test split on the data
│   │   │   │   ├── __init__.py
│   │   │   │   ├── nodes.py
│   │   │   │   ├── pipeline.py
│   │   │   │   ├── requirements.txt
│   │   │   │   └── README.md
│   │   │   ├── modelling: A pipeline that fits models, does hyperparameter search and reports on model performance
│   │   │   │   ├── __init__.py
│   │   │   │   ├── nodes.py
│   │   │   │   ├── pipeline.py
│   │   │   │   ├── requirements.txt
│   │   │   │   └── README.md
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   ├── pipeline.py: A master (or __default__) pipeline combines 3 modular pipelines from the above
│   │   └── run.py
│   ├── tests
│   │   ├── __init__.py
│   │   └── test_run.py
│   ├── requirements.txt
│   └── setup.py
├── .kedro.yml
├── README.md
├── kedro_cli.py
└── setup.cfg

What best practice should I follow to avoid leaking confidential data?
Avoid committing data to version control (data folder is by default ignored via .gitignore)
Avoid committing data to notebook output cells (data can easily sneak into notebooks when you don’t delete output cells)
Don’t commit sensitive results or plots to version control (in notebooks or otherwise)
Don’t commit credentials in conf/. There are two default folders for adding configuration - conf/base/ and conf/local/. Only the conf/local/ folder should be used for sensitive information like access credentials. To add credentials, please refer to the conf/base/credentials.yml file in the project template.
By default any file inside the conf/ folder (and its subfolders) containing credentials in its name will be ignored via .gitignore and not committed to your git repository.
To describe where your colleagues can access the credentials, you may edit the README.md to provide instructions.

# Ignoring notebook output cells in git
In order to automatically strip out all output cell contents before committing to git, you can run kedro activate-nbstripout. This will add a hook in .git/config which will run nbstripout before anything is committed to git.

Note: Your output cells will be left intact locally.


# examples
https://github.com/quantumblacklabs/kedro-examples
