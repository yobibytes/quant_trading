# Start jupyter

```
cmd (with admin rights)
wsl
cd /mnt/c/notebooks && jupyter lab --allow-root --ip=0.0.0.0 --no-browser
```

# Environment setup

```
sudo apt-get update && apt-get upgrade
sudo apt install build-essential wget -y
sudo apt-get install python3-dev python-pip python-dev python3 python python3-pip nodejs npm graphviz
sudo apt-get install libblas3 liblapack3 liblapack-dev libblas-dev gfortran
sudo pip install --upgrade pip
sudo pip install --upgrade setuptools
sudo pip install cython
sudo pip install pandas numpy numba scipy scikit-learn cachetools  psutil pyyaml requests python-dateutil statsmodels urllib3 beautifulsoup4 pandas-datareader xlrd python-dotenv ray munch
sudo pip install xgboost
sudo pip install tensorflow
sudo pip install findspark  fastparquet brotli pyarrow thrift
sudo pip install jupyter jupyterlab importlib seaborn matplotlib
sudo pip install cufflinks dash dash-renderer dash-html-components dash-core-components plotly chart-studio
sudo pip install joblib deap update_checker tqdm stopit dask[delayed] dask-ml scikit-mdr skrebate tpot
sudo pip install blaze sqlalchemy
sudo pip install graphviz featuretools
sudo pip install cvxpy pykalman cvxopt stats copulalib statistics
wget https://artiya4u.keybase.pub/TA-lib/ta-lib-0.4.0-src.tar.gz
tar -xvf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo pip install TA-Lib
sudo jupyter labextension install @jupyter-widgets/jupyterlab-manager
cd /mnt/c/notebooks && jupyter lab --allow-root --ip=0.0.0.0 --no-browser
```