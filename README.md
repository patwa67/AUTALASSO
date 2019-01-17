# AUTALASSO contains Julia code for the automatic adaptive LASSO presented in Waldmann et al. (2019; BMC Bioinformatics).
The code reads data from the supplied QTLMAS2010ny012.csv file (which needs to be downloaded to your woking directory and extracted). The y-variable (phenotype) is in the first column and the x-variables (SNPs; coded 0,1,2) are in the following columns (comma separated). The data is partitioned into training (generation 1-4) and test data (generation 5).

In order to run your own data, you need to have the same format as QTLMAS2010ny012.csv, unless you specify alternative options in the ```readdlm()``` function. You also need to specify which individuals to assign to training and test data, using the ... index of ```ytot[...]``` and ```X[...,2:size(X)[2]]``` (don't change the column index). No other changes are needed for a default run. 

If you want to perform cross-validation it is possible to use the ```randperm()``` function to generate training and test sets (for example for the first fold ```fold = 1```, ```cv1ind = randperm(MersenneTwister(fold), size(X)[1])``` and then ```ytrain = ytot[cv1ind[1:2326]]```, ```ytrain = ytot[cv1ind[2327:size(X)[1]]]```, ```Xtrain = X[cv1ind[1:2326],2:size(X)[2]]``` and ```Xtest= X[cv1ind[2327:size(X)[1]],2:size(X)[2]]```), repeat AUTALASSO over the folds with different result lists (for example changing res into ```res1```,...,```res5```) and take the average over the regression coefficients (```res1[2][1]```,...,```res5[2][1]```).
