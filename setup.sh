# create mamba environment and activate it
mamba create -n 07-failure-modes python
eval "$(conda shell.bash hook)"
conda activate 07-failure-modes

# install the ipython kernel for running jupyterlab
mamba install ipykernel ipywidgets

# for TAs to format the notebooks
# mamba install jupytext black nbconvert

# install libraries needed for the exercise
# model interpretability
pip install git+https://github.com/pytorch/captum.git
# computer vision deep learning
pip install torchvision
# progress bars
pip install tqdm
# scientific computing
pip install scipy
# machine learning
pip install scikit-learn
# data tables
pip install pandas
# visualization/graphing
pip install seaborn

# download data needed for the exercise
python download_mnist.py

conda deactivate