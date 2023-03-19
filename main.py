import matplotlib.pyplot as plt
import matplotlib.lines as mlines


from environment import Track


def main():
    # hyper parameters
    track_color = 'tab:blue'
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    track = Track(spacing=1.5)
    track.plot(ax=ax, color=track_color)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    mockup_line = mlines.Line2D([], [], color=track_color, label="Racing track")
    plt.legend(handles=[mockup_line])
    plt.show()


if __name__ == "__main__":
    main()
