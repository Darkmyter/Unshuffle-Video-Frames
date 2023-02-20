from utils import read_video, save_video
import numpy as np
import click


@click.command()
@click.option("--path", help="video path.")
@click.option("--output-path", help="video output path.")
@click.option("--seed", default=0, help="random seed.", type=int)
def shuffle(path, output_path, seed):

    frames = read_video(path)

    np.random.seed = seed
    np.random.shuffle(frames)

    save_video(range(len(frames)), frames, output_path)


if __name__ == "__main__":
    shuffle()