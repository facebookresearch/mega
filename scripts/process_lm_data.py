import os.path
import sys
import random


def remove_blank(fname, fwrite):
    with open(fname, "r", encoding='utf-8') as fin, open(fwrite, "w", encoding="utf-8") as fout:
        while True:
            title = fin.readline()
            if title == '':
                break
            while title.strip() == "":
                title = fin.readline()
            print(title)
            assert title.lstrip().startswith("= ")
            fout.write(title)
            blank_line = fin.readline()
            print(blank_line)
            assert blank_line.strip() == ""
            line = fin.readline()
            while line.strip() != "":
                fout.write(line)
                line = fin.readline()
            assert line.strip() == ""
            fout.write(line)


def random_sample(fname, fwrite, K=1000):
    counter = 0
    with open(fname, "r", encoding="utf-8") as fin, open(fwrite, "w", encoding="utf-8") as fout:
        while True:
            line = fin.readline()
            if line == '':
                break
            write = random.uniform(0, 1) > 0.5
            while line.strip() != '':
                if write:
                    fout.write(line)
                line = fin.readline()
            if write:
                fout.write(line)
            counter += 1 if write else 0
            if counter >= K:
                break


def process_wiki103():
    data_dir = "/private/home/chuntinz/checkpoint/research/lm/fairseq-apollo/data/wikitext-103"
    out_dir = "/private/home/chuntinz/checkpoint/research/lm/fairseq-apollo/data/wikitext-103-blank-removed"

    remove_blank(os.path.join(data_dir, "train.txt"), os.path.join(out_dir, "train.txt"))
    remove_blank(os.path.join(data_dir, "valid.txt"), os.path.join(out_dir, "valid.txt"))
    remove_blank(os.path.join(data_dir, "test.txt"), os.path.join(out_dir, "test.txt"))

    debug_dir = "/private/home/chuntinz/checkpoint/research/lm/fairseq-apollo/data/wt103-debug"
    random_sample(os.path.join(out_dir, "train.txt"), os.path.join(debug_dir, "train.txt"))
    random_sample(os.path.join(out_dir, "valid.txt"), os.path.join(debug_dir, "valid.txt"))
    random_sample(os.path.join(out_dir, "test.txt"), os.path.join(debug_dir, "test.txt"))


if __name__ == '__main__':
    process_wiki103()
