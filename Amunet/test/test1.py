import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import exp
import torch
import torch.nn.functional as F


if __name__ == '__main__':
    epoch=140
    para=15
    result = []
    for epoch in range(200):
        a = np.array(F.sigmoid(torch.tensor(12.5 - (epoch / 200) * para)))


        result.append(a)

    fig, ax = plt.subplots()
    ax.plot(range(200), result)

    ax.grid(True)
    plt.show()
    print('213')