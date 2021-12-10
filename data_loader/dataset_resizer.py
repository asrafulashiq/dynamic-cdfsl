"""Resize dataset
"""

import argparse
import os
from PIL import Image
import PIL
from tqdm import tqdm
from joblib import Parallel, delayed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", "-i", type=str, default=None)
    parser.add_argument("--out_path", "-o", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=20)
    args = parser.parse_args()

    in_path = os.path.expanduser(args.in_path)
    out_path = os.path.expanduser(args.out_path)

    assert in_path != out_path

    os.makedirs(out_path, exist_ok=True)

    def fn_loop(imfile_base):
        stem = os.path.splitext(imfile_base)[0]
        imfile = os.path.join(in_path, imfile_base)

        try:
            image = Image.open(imfile).convert('RGB')
        except PIL.UnidentifiedImageError:
            print(f"i/o error: {imfile} ")
            return
        image = image.resize((256, 256))
        image.save(os.path.join(out_path, stem + ".JPG"))

    Parallel(n_jobs=args.n_jobs)(delayed(fn_loop)(imfile_base)
                                 for imfile_base in tqdm(os.listdir(in_path)))
