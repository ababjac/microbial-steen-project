import os

# print('PCA-rhizo:')
# for i in range(5):
#     os.system('/usr/bin/time --format="Time: %E, Memory: %K" python3 scripts/PCA-Rhizo.py 2>> files/timing/rhizo.txt')
#
# print('Lasso-rhizo:')
# for i in range(5):
#     os.system('/usr/bin/time --format="Time: %E, Memory: %K" python3 scripts/Lasso-Rhizo.py 2>> files/timing/rhizo.txt')
#
# print('AE-rhizo:')
# for i in range(5):
#     os.system('/usr/bin/time --format="Time: %E, Memory: %K" python3 scripts/AE-Rhizo.py 2>> files/timing/rhizo.txt')

print('PCA-TARA:')
for i in range(5):
    os.system('/usr/bin/time --format="Time: %E, Memory: %K" python3 scripts/PCA-TARA.py 2>> files/timing/tara.txt')

print('Lasso-TARA:')
for i in range(5):
    os.system('/usr/bin/time --format="Time: %E, Memory: %K" python3 scripts/Lasso-TARA.py 2>> files/timing/tara.txt')

print('AE-TARA:')
for i in range(5):
    os.system('/usr/bin/time --format="Time: %E, Memory: %K" python3 scripts/AE-TARA.py 2>> files/timing/tara.txt')
