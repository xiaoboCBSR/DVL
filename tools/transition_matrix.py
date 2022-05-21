import numpy as np
import matplotlib.pyplot as plt

nb_classes = 5
type = 'pairflip' #'pairflip' #symmetric

if type == 'pairflip':
    # type1: pairflip
    P = np.eye(nb_classes)
    n = 0.45 #noise_rate
    P[0, 0], P[0, 1] = 1. - n, n
    for i in range(1, nb_classes - 1):
        P[i, i], P[i, i + 1] = 1. - n, n
    P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n
else:
    # type2: symmetric
    P = np.ones((nb_classes, nb_classes))
    n = 0.5 #noise_rate
    P = (n / (nb_classes - 1)) * P
    P[0, 0] = 1. - n
    for i in range(1, nb_classes - 1):
        P[i, i] = 1. - n
    P[nb_classes - 1, nb_classes - 1] = 1. - n

print (P)

plt.figure(figsize=(10, 10))
plt.matshow(P, fignum=0, cmap=plt.cm.Blues)  # gray, Blues, BrBG

for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        if P[i, j] != 0:
            if i==j:
                plt.text(x=j, y=i, s='%.0f%%' % (P[i, j]*100),
                         fontdict={'c': 'black', 'size': 22, 'weight': 'bold', 'family': 'monospace',
                                   'horizontalalignment': 'center', 'verticalalignment': 'center'})
            else:
                plt.text(x=j, y=i, s='%.1f%%' % (P[i, j] * 100),
                         fontdict={'c': 'black', 'size': 22, 'weight': 'bold', 'family': 'monospace',
                                   'horizontalalignment': 'center', 'verticalalignment': 'center'})
plt.show()


