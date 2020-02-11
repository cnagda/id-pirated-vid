import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


for dir in os.listdir("Temp"):
    if os.path.isdir(os.path.join(os.getcwd(), "Temp", dir)):
        legendlist = []
        plt.title(f"Testing Similarity of {dir}")
        plt.xlabel("Frame Number")
        plt.ylabel("Similarity Score")
        maxframes = 0
        for file in os.listdir(f"Temp/{dir}"):
            legendlist.append(file)
            fpath = os.path.join(os.getcwd(), "Temp", dir, file)
            with open(fpath) as f:
                lines = f.read().splitlines()
                numlines = [float(x) for x in lines]
                plt.plot(numlines)
                if len(lines) > maxframes:
                    maxframes = len(lines)

        xticks = list(np.linspace(0,maxframes-1,20,dtype=int))
        plt.xticks(xticks)
        plt.legend(legendlist,loc=0, prop={'size': 6})
        plt.show()

# TODO: delete Temp dir but I don't want to do that yet
