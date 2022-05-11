import os

#progs = ['AE-RF-Rhizo', 'Lasso-RF-Rhizo', 'PCA-RF-Rhizo', 'tSNE-RF-Rhizo', 'AE-RF-Malaria', 'Lasso-RF-Malaria', 'PCA-RF-Malaria', 'tSNE-RF-Malaria', 'AE-RF-TARA', 'Lasso-RF-TARA', 'PCA-RF-TARA', 'tSNE-RF-TARA', 'AE-RF-GEM', 'Lasso-RF-GEM', 'PCA-RF-GEM', 'tSNE-RF-GEM']
progs = ['AE-SVM-Rhizo', 'Lasso-SVM-Rhizo', 'PCA-SVM-Rhizo', 'tSNE-SVM-Rhizo', 'AE-SVM-Malaria', 'Lasso-SVM-Malaria', 'PCA-SVM-Malaria', 'tSNE-SVM-Malaria', 'AE-SVM-TARA', 'Lasso-SVM-TARA', 'PCA-SVM-TARA', 'tSNE-SVM-TARA', 'AE-SVM-GEM', 'Lasso-SVM-GEM', 'PCA-SVM-GEM', 'tSNE-SVM-GEM']

for prog_name in progs:
    command = 'mprof run -o files/memory/'+prog_name+'.dat scripts/'+prog_name+'.py'
    os.system(command)
