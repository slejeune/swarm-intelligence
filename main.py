import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as ani
import time
import numpy as np

from environment import Track


def main():
    # hyper parameters
    track_color = "tab:blue"
    boid_marker = ">"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    track = Track(spacing=1)
    
    # N = 15, 20, 25, 30 and 34
    x, y = track.sample(size=100)

    xlist = np.linspace(-6.0, 6.0, 1000)
    ylist = np.linspace(-6.0, 6.0, 1000)
    X, Y = np.meshgrid(xlist, ylist)
    Z = (Track.border(X, Y, 2, 1.45, 1) >= 1) & (Track.border(X, Y, 2, 1.45 + 0.8, 1 + 1.2) <= 1)
    cp = ax.contourf(X, Y, Z)

    scatter = plt.scatter(x, y, marker=boid_marker)

    def animate(i=int):
        positions, markers = track.update()
        scatter.set_offsets(positions)
        paths = [m.get_path().transformed(m.get_transform()) for m in markers]
        scatter.set_paths(paths)
        return scatter

    track.plot(ax=ax, color=track_color)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    mockup_line = mlines.Line2D([], [], color=track_color, label="Racing track")
    mockup_boid = mlines.Line2D(
        [], [], color=track_color, label="Boid", marker=boid_marker, linestyle="None", markersize=10,
    )
    plt.legend(handles=[mockup_line, mockup_boid])

    animator = ani.FuncAnimation(fig, animate, interval=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
