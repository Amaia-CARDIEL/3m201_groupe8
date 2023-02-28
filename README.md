### Basic MNIST classification

This repository contains the undergraduate research project, my team and I did to classify MNIST images in Spring 2021. 
The involved team members were: Aya Bouzidi, Camille Grimal, Elysé Rasoloarivony and Amaia Cardiel (myself). 

We explored three Machine Learning algorithms to perform MNIST classification:

  * Naive classification via euclidian distance to class centroids
  * Classificationvia Singular Value Decomposition (SVD)
  * Classification via Tangent Distance

This last method was implemented, following the paper [Transformation invariance in pattern recognition - tangent distance and tangent propagation](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_17).

### A few words on some of the files

***Rapport_final.pdf*** is the final report we submitted regarding this research project

***base.py*** gathers the basic structures that we would need for any project

***exceptions.py*** gathers the various errors that might arise

***mnist-original.mat*** is the original dataset that was provided to us by our academic supervisor

***base_apprentissage.mat*** and ***base_tests.mat*** are datasets in the same format as the file *mnist-original.mat* and can thus be used in the same way. Their title are explicit regarding their content: *base_apprentissage.mat* is meant for training (80% of the original dataset) and *base_tests.mat* for testing (20% of the original dataset). 

