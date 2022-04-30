import os

# print('PCA-rhizo:')
# for i in range(5):
#     os.system('/usr/bin/time --append --output files/timing/rhizo.txt --format="Time: %E, Memory: %K" python3 scripts/PCA-SVM-Rhizo.py')
#
# print('Lasso-rhizo:')
# for i in range(5):
#     os.system('/usr/bin/time --append --output files/timing/rhizo.txt --format="Time: %E, Memory: %K" python3 scripts/Lasso-SVM-Rhizo.py')

print('tSNE-rhizo:')
for i in range(5):
     os.system('/usr/bin/time --append --output files/timing/rhizo.txt --format="Time: %E, Memory: %K" python3 scripts/tSNE-SVM-Rhizo.py')
#
# print('AE-rhizo:')
# for i in range(5):
#     os.system('/usr/bin/time --append --output files/timing/rhizo.txt --format="Time: %E, Memory: %K" python3 scripts/AE-SVM-Rhizo.py')

# print('PCA-TARA:')
# for i in range(5):
#     os.system('/usr/bin/time --append --output files/timing/tara.txt --format="Time: %E, Memory: %K" python3 scripts/PCA-SVM-TARA.py')
#
# print('Lasso-TARA:')
# for i in range(5):
#     os.system('/usr/bin/time --append --output files/timing/tara.txt --format="Time: %E, Memory: %K" python3 scripts/Lasso-SVM-TARA.py')

print('tSNE-TARA:')
for i in range(5):
     os.system('/usr/bin/time --append --output files/timing/tara.txt --format="Time: %E, Memory: %K" python3 scripts/tSNE-SVM-TARA.py')
#
# print('AE-TARA:')
# for i in range(5):
#     os.system('/usr/bin/time --append --output files/timing/tara.txt --format="Time: %E, Memory: %K" python3 scripts/AE-SVM-TARA.py')


print('PCA-rhizo:')
for i in range(5):
    os.system('/usr/bin/time --append --output files/timing/rhizo-RF.txt --format="Time: %E, Memory: %K" python3 scripts/PCA-RF-Rhizo.py')

print('Lasso-rhizo:')
for i in range(5):
    os.system('/usr/bin/time --append --output files/timing/rhizo-RF.txt --format="Time: %E, Memory: %K" python3 scripts/Lasso-RF-Rhizo.py')

print('tSNE-rhizo:')
for i in range(5):
     os.system('/usr/bin/time --append --output files/timing/rhizo-RF.txt --format="Time: %E, Memory: %K" python3 scripts/tSNE-RF-Rhizo.py')

print('AE-rhizo:')
for i in range(5):
    os.system('/usr/bin/time --append --output files/timing/rhizo-RF.txt --format="Time: %E, Memory: %K" python3 scripts/AE-RF-Rhizo.py')

print('PCA-TARA:')
for i in range(5):
    os.system('/usr/bin/time --append --output files/timing/tara-RF.txt --format="Time: %E, Memory: %K" python3 scripts/PCA-RF-TARA.py')

print('Lasso-TARA:')
for i in range(5):
    os.system('/usr/bin/time --append --output files/timing/tara-RF.txt --format="Time: %E, Memory: %K" python3 scripts/Lasso-RF-TARA.py')

print('tSNE-TARA:')
for i in range(5):
     os.system('/usr/bin/time --append --output files/timing/tara-RF.txt --format="Time: %E, Memory: %K" python3 scripts/tSNE-RF-TARA.py')

print('AE-TARA:')
for i in range(5):
    os.system('/usr/bin/time --append --output files/timing/tara-RF.txt --format="Time: %E, Memory: %K" python3 scripts/AE-RF-TARA.py')
