import sys

labeled_set_name = sys.argv[1]
with open(labeled_set_name) as f:
    labed_lines = f.readlines()

n_samples = len(labed_lines)/4
n_cut = round(n_samples * 0.8)

labed_lines_train = labed_lines[:n_cut*4]
labed_lines_validate = labed_lines[n_cut*4:]

with open(labeled_set_name[:-4] + "_train.txt", "w") as f:
    f.writelines(labed_lines_train)

with open(labeled_set_name[:-4] + "_validate.txt", "w") as f:
    f.writelines(labed_lines_validate)

