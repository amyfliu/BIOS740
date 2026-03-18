## BIOS740 - HW04

Generative Adversarial Networks (GANs) represent a powerful class of neural networks capable of producing high-fidelity synthetic data. First introduced by Goodfellow et al. (2014), a GAN typically comprises two competing models: a Generator (G) and a Discriminator (D). The generator strives to create realistic samples that mimic the distribution of real data, while the discriminator aims to distinguish between these synthetic (“fake”) samples and genuine (“real”) data. Through this adversarial process, both models iteratively improve, leading to the production of increasingly realistic synthetic data.

In this assignment, you will develop a GAN to generate synthetic samples based on the given dataset (T1 and T2 image slices in PNG format). The goal is to 
-	(1) build a deeper understanding of GAN architectures and training procedures, 
-	(2) learn how to evaluate synthetic data quality, and 
-	(3) discuss the potential benefits and limitations of using GANs in real-world scenarios, such as data augmentation or privacy preservation.

### Data
We'll work with a MRI Image dataset with ~20K T1 and T2 image slices in PNG format. 

### Assignment Details

Please refer to the **GAN.ipynb** notebook for the completed implementation and detailed results. 