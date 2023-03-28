import json
import matplotlib.pyplot as plt
import itertools

marker = itertools.cycle(('s', 'o', 'D', '>', '*')) 

sizes = [15, 20, 25, 30, 34]

fig, ax = plt.subplots()

for size in sizes:
    with open(rf"./results_{size}.json", 'r') as f:
        results = json.load(f)

    ax.scatter(results['density'], results['speed'], label=f"N={size}", marker=next(marker))
ax.set_xlabel("density")
ax.set_ylabel("speed")
ax.set_title("Fundamental diagram for boids on a circular track\nAlignment=False, Cohesion=False, Separation=False")
ax.legend(title="population size")
fig.tight_layout()
plt.show()