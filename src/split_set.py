import sys
import random

labeled_set_name = sys.argv[1]
with open(labeled_set_name) as f:
    labed_lines = f.readlines()

n_samples = len(labed_lines)/4
n_cut = round(n_samples * 0.8) # 80% 

labled_lines_validate = labed_lines[n_cut*4:]
labled_lines_train = labed_lines[:n_cut*4]
test_lines = [4*random.randrange(n_cut) for n in range(200)]

test_set = []
for n in test_lines:
    test_set.append(labled_lines_train.pop(n))
    test_set.append(labled_lines_train.pop(n))
    test_set.append(labled_lines_train.pop(n))
    test_set.append(labled_lines_train.pop(n))

with open(labeled_set_name[:-4] + "_train.txt", "w") as f:
    f.writelines(labled_lines_train)

with open(labeled_set_name[:-4] + "_validate.txt", "w") as f:
    f.writelines(labled_lines_validate)

with open("test_set.txt", "w") as f:
    f.writelines(test_set)



