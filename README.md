<b>Curiosity Driven Exploration to Optimize Structure-Property Learning in Microscopy</b>

Aditya Vatsavai<sup>1,2</sup> Ganesh Narasimha<sup>1</sup>, Yongtao Liu<sup>1</sup>, Jan-Chi Yang<sup>3</sup>, Hiroshi Funakubo<sup>4</sup>, Maxim Ziatdinov<sup>5</sup> and Rama Vasudevan<sup>1</sup>

1 Center for Nanophase Materials Sciences, Oak Ridge National Laboratory, Oak Ridge, TN, USA - 37831
2 Department of Physics, University of North Carolina, Chapel Hill
3 Department of Physics, National Cheng Kung University, Tainan 70101, Taiwan
4 Department of Materials Science and Engineering, Tokyo Institute of Technology, Yokohama, 226-8502, Japan
5 Physical Sciences Division, Pacific Northwest National Laboratory (PNNL), Richland, Washington, USA – 99352}

This repository contains code for the curiosity-driven exploration algorithms for the above paper.

The Data files are located in 'Data Files'. The raw data files will be published with the paper.

There are 3 notebooks:

(1) im2spec_enoded_error_model.ipynb: In this notebook, we show the first version of the algorithm which uses an initial ensemble of im2spec models for training and spectral reconstruction. The latent embeddings are then used to train an error model. This error estimation methodology is used for active learning in Scanning Probe Microscopy(SPM)

(2) Curiosity_sampling.ipynb: This notebook details the second algorithm described in the paper, where an error model is trained on the latent space encodings of an im2spec model.

(3) Curiosity_Algorithm_on_Microscope: This notebook runs the algorithm actively on the microscope, using our AEcroscopy suite.

In addition, for benchmarking purposes MPI Code is provided to test the performance on pre-acquired data on multiple processors in parallel. 
