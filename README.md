# StellaRGCN
Custom RGCN architecture for classifying stellar objects (Star/Galaxy/QSO) from SDSS telescope data using photometric features and redshift. Built with PyTorch Geometric, Scikit-learn &amp; Tableau.

The project involves a deep learning model for identifying the type of stellar object, such as Star, Galaxy, and Quasar (QSO), as seen by ground-based telescopes based on photometric and spectroscopic characteristics.
While traditional classification models use hand-crafted thresholds and shallow ML models, this project takes a step further by introducing the concept of inter-object relationships by designing a custom model called Relational Graph Convolutional Network (RGCN), allowing the model to take advantage of structural similarities between objects in feature space.
A Residual MLP ensemble with Stochastic Weight Averaging (SWA) is also implemented as a strong baseline.
The dataset used for this project is the Sloan Digital Sky Survey (SDSS) stellar classification dataset, which contains ~100,000 observations.


Key Features

Custom RGCN architecture utilizing PyTorch Geometric for node-level prediction of stellar objects
Residual MLP ensemble with 5 seeds, SWA, and cosine annealing as a high-performing baseline
Graph construction from tabular telescope data, with nodes as observations and edges between similar objects
Class-weighted loss with label smoothing for class imbalance
Use of Test-Time Augmentation as an ensemble method
Tableau dashboard creation for EDA of photometric features and redshift distributions

StellaRGCN/
│
├── Star.ipynb                  # Main notebook: RGCN + Residual MLP pipeline
├── star_classification.csv     # SDSS dataset (~100k observations)
├── Star_Model.pth              # Saved model weights
├── README.md

Dataset
Source: Sloan Digital Sky Survey (SDSS) — star_classification.csv

Feature                       Description

alpha, delta                  Right Ascension & Declination (sky coordinates)
u,g,r,i,z                     Photometric filter magnitudes (ultraviolet → infrared)
redshift                      Spectroscopic redshift — key discriminator for QSOs
run_ID,cam_col,field_ID       Observation metadata
plate,MJD,fiber_ID            Spectrograph identifiers
class                         Target label: STAR,GALAXY,QSO

Size: ~100,000 rows
Classes: 3 (STAR, GALAXY, QSO)

Model Architecture

1. Relational GCN (RGCN)

Nodes represent individual telescope observations (feature vectors of photometric bands + redshift)
Edges are constructed based on feature-space proximity (k-NN graph)
Relation types encode different edge semantics (e.g., same sky region, similar magnitude profile)
Multiple RGCN layers with ReLU activations and dropout
Final softmax layer over 3 output classes


2. Residual MLP Ensemble (Baseline)

Multi-layer perceptron with residual skip connections
5-seed ensemble training for variance reduction
Stochastic Weight Averaging (SWA) with cosine annealing warm restarts
AdamW optimiser with weight decay, gradient clipping
Early stopping (patience = 50 epochs, max 500 epochs)
Batch size: 1024 | Label smoothing: 0.02


Getting Started

Prerequisites

pip install torch torchvision torchaudio
pip install torch_geometric
pip install scikit-learn pandas numpy matplotlib
 Tested with Python 3.10+, PyTorch 2.x, CUDA 12.6 (GPU recommended).

Running the Notebook
 Clone the repository:
 git clone https://github.com/Shashank7490/StellaRGCN.git
   cd StellaRGCN
   
Place star_classification.csv in the root directory.
Open and run Star.ipynb end-to-end in Jupyter or Google Colab.

To load the pre-trained model:

import torch
   model = torch.load('Star_Model.pth')
   model.eval()

Results

Model                             Test Accuracy

Residual MLP Ensemble (SWA)        ~98%+
RGCN                               See notebook for latest run

 #Results may vary slightly depending on random seed and hardware.

Visualisations

Exploratory data analysis and feature distributions were visualised using Tableau, including:
Redshift distributions by object class
Photometric magnitude spread across SDSS filters (u, g, r, i, z)
Sky coordinate (RA/Dec) scatter plots coloured by object type


Tech Stack

Python 3.10+,
PyTorch + PyTorch Geometric,
Scikit-learn,
NumPy / Pandas,
Tableau,
Google Colab (T4 GPU)

Duration: October 2025 

License
This project is licensed under the MIT License

Acknowledgements

Sloan Digital Sky Survey (SDSS) for the stellar classification dataset
PyTorch Geometric for the GNN framework
Kaggle community for dataset hosting
