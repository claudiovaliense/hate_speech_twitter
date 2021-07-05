#Solution presented in the Hate Speech Twitter competition task at PAN @ CLEF 2021

# Tutorial execution
Create a virtual environment to install all packages in their specific versions, which ensures greater success in executing the algorithm.

# Clone project
git clone https://github.com/claudiovaliense/hate_speech_twitter.git

# Project root
cd hate\_speech\_twitter

# Install virtual enviroment and python 3.6
sudo apt-get install virtualenv && sudo apt-get install python3.6

# Create virtual environment
virtualenv -p python3.6 virtualenv

# Ativate virtual environment
source virtualenv/bin/activate

# Install package and version specific
pip install -r requeriments.txt

# Create representation the texts, example  of the representation in file .sh
sh representations.py.sh

# Optimize hyper-parameter of classifier
sh classifier\_score.py.sh

# Train and predict the classes 
sh classifiers\_predict.py.sh

# Print statistics of results, save stats in stats folder
sh classifiers\_statistics.py.sh

# Expected result at the end of the process
pan21-author-profiling-training-2021-03-142en\_word\_tfidf\_classifier\_svm & 66.32$\pm$9.18\\

# To run for the other representations, just generate all representations through the representations.py.sh file

# This file contains information on how the splitting of the folds was generated, to evaluate the method.  pan2021.py 
