import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_excel('../bending-modulus.xlsx')

y='Lx_0, nm'
x_vars=['kappa, kT (q^-4)']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for x in x_vars:
    tmp=df.sort_values(by=x)
    Y=tmp[y]
    X=tmp[x]

    axes[0].plot(X, Y, linestyle='--', marker='o', label=x)
    axes[0].legend()

    axes[1].scatter(X,Y,label=x)
    axes[1].legend()

plt.show()
