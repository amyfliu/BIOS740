## BIOS740 - HW02

This assignment consists of two parts designed to deepen your understanding of sequence modeling. The first part focuses on classifying heartbeat signals using RNN, LSTM, and GRU. The second part is based on a step-by-step RNN implementation adapted from Module 5 of Andrew Ng's Deep Learning course.

#### Part 1: Heartbeat Classification

You will work with heartbeat signals from the MIT-BIH Arrhythmia Dataset. Each row in the dataset represents a time series with 187 time steps and one feature per time step. The last column is the label, indicating one of five classes: 'N' (Normal): 0; 'S' (Supraventricular premature beat): 1; 'V' (Premature ventricular contraction): 2; 'F' (Fusion of ventricular and normal beat): 3; 'Q' (Unclassifiable beat): 4

#### Part 2: Building an RNN Step-by-Step**

This part of the assignment is adapted from Module 5 of Andrew Ng's Deep Learning course. You will build an RNN step-by-step to better understand the internal workings of sequence models.

### PLEASE READ!

**The overall workflow for the project is as follows:**

1.  Testing and debugging of all code via jupyter notebook via Longleaf OnDemand; testing using 1 epoch (`hw2_part1_heartbeat_classification_assignment.ipynb`). 
2.  Jupyter Notebook (GPU-FULL) with full A100 GPU and 40GB GPU memory

**Key files**

-   `hw2_part1_heartbeat_classification_assignment.ipynb`: complete code and write up for the assignment
-   `first_feature.png`: example of what the first channel looks like for a particular sample
