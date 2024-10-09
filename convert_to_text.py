import glob
import os
import numpy as np
from tqdm import tqdm


files = glob.glob("./scores/**/*.npy", recursive=True)

for file in tqdm(files):
    x = np.load(file)
    textfilename = file.replace(".npy", ".txt")
    textfilename = textfilename.replace("scores", "txtscores")
    os.makedirs(os.path.split(textfilename)[0], exist_ok=True)
    np.savetxt(textfilename, x)
