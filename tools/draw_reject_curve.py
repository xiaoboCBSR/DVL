import matplotlib.pyplot as plt
import numpy as np


X = [i for i in np.arange(0.0, 1.0, 0.05)]
DVL_free = [0.9101, 0.9405, 0.9590, 0.9736, 0.9849, 0.9890, 0.9923, 0.9944,
            0.9957, 0.9970, 0.9972, 0.9982, 0.9986, 0.9986, 0.9987, 0.9991,
            0.9991, 1.0, 1.0, 1.0]
Sherlock_free = [0.8738, 0.9030, 0.9251, 0.9405, 0.9517, 0.9615, 0.9687, 0.9742,
                 0.9786, 0.9808, 0.9840, 0.9861, 0.9875, 0.9884, 0.9889, 0.9885,
                 0.9888, 0.9876, 0.9849, 0.9775]
SATO_free = [0.9031, 0.9310, 0.9503, 0.9635, 0.9718, 0.9782, 0.9834, 0.9871,
             0.9888, 0.9899, 0.9911, 0.9914, 0.9918, 0.9923, 0.9930, 0.9929,
             0.9925, 0.9917, 0.9894, 0.9862]

plt.figure(figsize=(13, 8.5))
plt.grid()
plt.xlim((0.0, 1.0))
plt.ylim((0.85, 1.01))
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.xlabel("Fraction of Samples Rejected", fontsize=20, fontweight='bold')
plt.ylabel("Support-weighted F1 score", fontsize=20, fontweight='bold')

plt.plot(X, Sherlock_free, color='g', label="Sherlock", linewidth=4.0)
plt.plot(X, SATO_free, color='b', label="SATO_NoStruct", linewidth=4.0)
plt.plot(X, DVL_free, color='r', label="DVL", linewidth=4.0)
plt.legend(loc='lower right', prop={'size': 20.0})
plt.show()