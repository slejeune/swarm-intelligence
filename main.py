import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as ani
import time

from environment import Track


def main():
    # hyper parameters
    track_color = "tab:blue"
    boid_marker = ">"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    track = Track(spacing=0.1)
    x, y = track.sample(size=1)

    scatter = plt.scatter(x, y, marker=boid_marker)

    def animate(i=int):
        positions, _ = track.update()
        scatter.set_offsets(positions)

        # scatter.set_paths(markers)
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
    plt.show()

if __name__ == "__main__":
    main()
