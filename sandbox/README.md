# Start jupyter

```cmd (with admin rights)
wsl
sudo mkdir -p /mnt/d
sudo mount -t drvfs d: /mnt/d && cd /mnt/d/notebooks && jupyter lab --allow-root --ip=0.0.0.0 --no-browser
```

# Conda environment
```sh (within wsl)

# init
conda --version
conda update conda
conda config --add channels conda-forge

# setup env
echo $CONDA_PREFIX
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#    #!/bin/sh
#    export MY_KEY='secret-key-value'
#    export MY_FILE=/path/to/my/file/
touch $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
#    #!/bin/sh
#    unset MY_KEY
#    unset MY_FILE


# create
conda create --name py38 python=3.8
# or: conda create --prefix ./envs/test python=3.8
conda info --envs

# save & load
conda env export --from-history > environment.yml
conda env create -f environment.yml
conda list --explicit > py38.txt
conda env create --file py38.txt 

# update
conda env update --prefix ./env --file environment.yml  --prune

# search packages
conda search PACKAGE # https://docs.anaconda.com/anaconda/packages/pkg-docs/

# usage
source activate py38
# or: source activate ./envs/test
source deactivate

# delete env
conda remove --name py38 --all
```

# Create link
```cmd
mklink d:\notebooks\sandbox\model\base d:\notebooks\sandbox\model\20200107
```

```
import platform;
print(platform.architecture());
```

# (Optional) Reboot WSL

# CMD (admin)
```cmd
net stop LxssManager
net start LxssManager
```

# Environment setup

```power-shell (with admin rights)
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl -l -v
wsl --set-version <distro> 2
wsl --set-default-version 2
``` 
```wsl2 configuration
# https://docs.microsoft.com/en-us/windows/wsl/release-notes#build-18945
[wsl2]
kernel=<path>              # An absolute Windows path to a custom Linux kernel.
memory=<size>              # How much memory to assign to the WSL2 VM.
processors=<number>        # How many processors to assign to the WSL2 VM.
swap=<size>                # How much swap space to add to the WSL2 VM. 0 for no swap file.
swapFile=<path>            # An absolute Windows path to the swap vhd.
localhostForwarding=<bool> # Boolean specifying if ports bound to wildcard or localhost in the WSL2 VM should be connectable from the host via localhost:port (default true).
# <path> entries must be absolute Windows paths with escaped backslashes, for example C:\\Users\\Ben\\kernel
# <size> entries must be size followed by unit, for example 8GB or 512MB
```
```powershell (with admin rights)
Invoke-WebRequest -Uri https://aka.ms/wsl-ubuntu-1804 -OutFile Ubuntu.zip -UseBasicParsing
# Rename-Item ./Ubuntu.appx ./Ubuntu.zip
Expand-Archive ./Ubuntu.zip ~/wsl/Ubuntu-1804-wsl
~/wsl/Ubuntu-1804-wsl/Ubuntu1804.exe
# add ~/wsl/Ubuntu-1804-wsl to PATH
wslconfig /list
wsl -l -v
wsl --set-default <distro>
```
* https://docs.docker.com/docker-for-windows/wsl-tech-preview/
* https://code.visualstudio.com/download#
* https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl
JDK
IntelliJ
pycharm
sourcetree
KeePass

```
sudo apt-get update && sudo apt-get upgrade
sudo apt install build-essential wget -y
sudo apt-get install python3-dev python-pip python-dev python3 python python3-pip nodejs npm graphviz
sudo apt-get install libblas3 liblapack3 liblapack-dev libblas-dev gfortran
sudo apt-get install libatlas-base-dev python-dev gfortran pkg-config libfreetype6-dev
sudo apt-get install default-jdk
sudo pip install --upgrade pip
sudo pip install --upgrade setuptools
sudo pip install -U cython
sudo pip install -U python-dotenv pandas numpy numba scipy scikit-learn cachetools psutil pyyaml requests python-dateutil statsmodels urllib3 beautifulsoup4 pandas-datareader xlrd python-dotenv munch openpyxl==3.0.1
sudo pip install -U ray
sudo pip install -U xgboost
sudo pip install -U tensorflow
sudo pip install -U findspark  fastparquet brotli pyarrow thrift
sudo pip install -U jupyter jupyterlab importlib seaborn matplotlib jupyter_server_proxy jupyterlab-dash
sudo pip install -U jupyter_contrib_nbextensions
sudo pip install -U qgrid
sudo pip install -U quandl
sudo pip install -U html5lib
sudo pip install -U bs4
sudo pip install -U yfinance
sudo pip install -U investpy
sudo pip install -U requests_cache
sudo pip install -U mpl_finance
sudo pip install -U --user PyYAML
sudo pip install -U kedro kedro-viz kedro-airflow kedro-docker
sudo pip install -U apache-airflow
sudo pip install -U eli5
sudo pip install -U umap-learn hdbscan
sudo jupyter labextension install jupyterlab-dash
sudo jupyter labextension install jupyter_server_proxy
sudo jupyter labextension install @jupyter-widgets/jupyterlab-manager
sudo jupyter serverextension enable --py jupyterlab --sys-prefix
sudo jupyter serverextension enable --py jupyter_server_proxy --sys-prefix
sudo jupyter labextension install qgrid
sudo pip install -U cufflinks dash dash-renderer dash-html-components dash-core-components plotly chart-studio dash-daq dash-cytoscape gunicorn dash-bootstrap-components plotly-express
sudo pip install -U joblib deap update_checker tqdm stopit dask[delayed] dask-ml scikit-mdr skrebate tpot
sudo pip install -U blaze sqlalchemy
sudo pip install -U alphalens pyfolio holidays
sudo pip install -U luigi apache-airflow[all]
sudo pip install -U torch
sudo pip install -U Pillow
sudo pip install -U wheel
sudo pip install -U memory_profiler
sudo pip install -U pysnooper
sudo pip install -U mlflow
sudo pip install -U category_encoders sklearn_pandas
sudo pip install -U pygit2
sudo pip install -U requests tabulate "colorama>=0.3.8" future
sudo pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
# deprecated
# sudo pip install empyrical
# requires python 3.5
# sudo pip install zipline
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
sudo -E jupyter contrib nbextension install --sys-prefix

sudo -E jupyter nbextension enable scratchpad/main --sys-prefix
jupyter notebook --generate-config
jupyter notebook password
cd /mnt/c/notebooks && jupyter lab --allow-root --ip=0.0.0.0 --no-browser

# install anaconda: https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
ANACONDA_INSTALLER=Anaconda3-2020.02-Linux-x86_64.sh
cd /tmp
curl -O https://repo.anaconda.com/archive/$ANACONDA_INSTALLER
sha256sum $ANACONDA_INSTALLER
bash $ANACONDA_INSTALLER
source ~/.bashrc
conda list
```


# fix for yahoo finance
pd.core.common.is_list_like = pd.api.types.is_list_like


# WSL VHDX disk size
* https://docs.microsoft.com/en-us/windows/wsl/wsl2-ux-changes#understanding-wsl-2-uses-a-vhd-and-what-to-do-if-you-reach-its-max-size