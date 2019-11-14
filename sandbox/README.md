# Start jupyter

```cmd (with admin rights)
wsl
cd /mnt/c/notebooks && jupyter lab --allow-root --ip=0.0.0.0 --no-browser
```

# Environment setup

```power-shell (with admin rights)
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
Invoke-WebRequest -Uri https://aka.ms/wsl-ubuntu-1804 -OutFile Ubuntu.zip -UseBasicParsing
# Rename-Item ./Ubuntu.appx ./Ubuntu.zip
Expand-Archive ./Ubuntu.zip ~/wsl/Ubuntu-1804-wsl
~/wsl/Ubuntu-1804-wsl/Ubuntu1804.exe
# add ~/wsl/Ubuntu-1804-wsl to PATH
wslconfig /list
``` 

```
sudo apt-get update && apt-get upgrade
sudo apt install build-essential wget -y
sudo apt-get install python3-dev python-pip python-dev python3 python python3-pip nodejs npm graphviz
sudo apt-get install libblas3 liblapack3 liblapack-dev libblas-dev gfortran
sudo pip install --upgrade pip
sudo pip install --upgrade setuptools
sudo pip install cython
sudo pip install python-dotenv pandas numpy numba scipy scikit-learn cachetools  psutil pyyaml requests python-dateutil statsmodels urllib3 beautifulsoup4 pandas-datareader xlrd python-dotenv munch
sudo pip install ray
sudo pip install xgboost
sudo pip install tensorflow
sudo pip install findspark  fastparquet brotli pyarrow thrift
sudo pip install jupyter jupyterlab importlib seaborn matplotlib jupyter_server_proxy jupyterlab-dash
sudo jupyter labextension install jupyterlab-dash
sudo jupyter labextension install jupyter_server_proxy
sudo jupyter serverextension enable --py jupyterlab --sys-prefix
sudo jupyter serverextension enable --py jupyter_server_proxy --sys-prefix
sudo pip install cufflinks dash dash-renderer dash-html-components dash-core-components plotly chart-studio dash-daq dash-cytoscape gunicorn dash-bootstrap-components plotly-express
sudo pip install joblib deap update_checker tqdm stopit dask[delayed] dask-ml scikit-mdr skrebate tpot
sudo pip install blaze sqlalchemy
sudo pip install graphviz featuretools
sudo pip install cvxpy pykalman cvxopt stats copulalib statistics
sudo pip install nltk
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