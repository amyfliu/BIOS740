## BIOS740 - HW#1

### Background

Alzheimer's Disease (AD) is a neurodegenerative disorder characterized by progressive cognitive decline. Atrophy in the hippocampus, a brain region critical for memory formation, is one of the early biomarkers of AD. In this assignment, you are provided with left and right hippocampus matrices in compressed CSV format for 369 subjects from ADNI. Each matrix represents imaging data preprocessed and extracted from MRI scans. Your task is to develop a Convolutional Neural Network (CNN) model to predict the AD status of subjects based on their hippocampal data.

### PLEASE READ!

**The overall pipeline for the project is as follows:**

1.  Testing and debugging of all code via jupyter notebook via Longleaf OnDemand; testing using 1 epoch (`HW1_v3.ipynb`)

2.  Convert notebook to a .py script (jupyter nbconvert --to script HW1_v3.ipynb) --\> `HW1_v3.py`

3.  Make SLURM submission script (run_hw1_test.sbatch)

4.  Submit the job (sbatch run_hw1_test.sbatch)

**Key files**

-   `HW1_v3.ipynb`: complete code and write up for the assignment

-   `HW1_v3.py`: identical to above except in .py format; needed for slurm

-   `run_hw1_test.sbatch`: slurm file used to submit the job

-   `first_feature.png`: example of what the first channel looks like for a particular sample

-   `all_channels.png`: example of what 14 channels look like; note the similarities between left and right hippocampus

-   `loss_curve.png`: shows training and validation loss over the epochs

-   `hw1_simpleCNN.csv`: predictions of AD status on testing set (using CNN)

-   `hw1_vgg.csv`: prediction of AD status on testing set (using VGG)

**NOTE: ** GPU wait time is taking unexpectedly long time (~1 Day in queue). There will be minor updates to the final .csv prediction files once the full job is complete.


