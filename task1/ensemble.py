import sys
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import json

prefix = sys.argv[1]+"_"
dir_path = sys.argv[1]
files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

d = {}
for f_path in files:
    print("> reading: ", dir_path + "/" + f_path)
    f = open(dir_path + "/" + f_path)
    data = json.load(f)
    for key in data:
        if len(data[key]) == 2:
            start, end = data[key]
            if key not in d:
                d[key] = []
            d[key].append([start, end])
        assert len(d[key]) > 0

ensemble = {} # find the majority of the pair, exact match
ensemble_start_end_separate = {} # find the majority of start and end independently
ensemble_start = {} # find the majority of start and take the last end
ensemble_start_sort_end = {} # find the majority of start and take the most frequent end

# voting
for key in d:
    counter = defaultdict(list)
    majority = None
    max_count = 0

    for i in range(len(d[key])):
        if str(d[key][i]) not in counter:
            counter[str(d[key][i])] = 1
        else:
            counter[str(d[key][i])] += 1
        if max_count < counter[str(d[key][i])]:
            max_count = counter[str(d[key][i])]
            majority = d[key][i]
    # print(d[key])
    # print(max_count)
    ensemble[key] = majority

for key in d:
    counter_start = {}
    counter_end = {}
    majority_start = None
    majority_end = None
    max_count_start = 0
    max_count_end = 0

    for i in range(len(d[key])):
        if d[key][i][0] not in counter_start:
            counter_start[d[key][i][0]] = 1
        else:
            counter_start[d[key][i][0]] += 1
        if max_count_start < counter_start[d[key][i][0]]:
            max_count_start = counter_start[d[key][i][0]]
            majority_start = d[key][i][0]

        if d[key][i][1] not in counter_end:
            counter_end[d[key][i][1]] = 1
        else:
            counter_end[d[key][i][1]] += 1
        if max_count_end < counter_end[d[key][i][1]]:
            max_count_end = counter_end[d[key][i][1]]
            majority_end = d[key][i][1]

    if majority_end < majority_start:
        # print("ohno")
        ensemble_start_end_separate[key] = ensemble[key]
    else:
        ensemble_start_end_separate[key] = [majority_start, majority_end]

for key in d:
    counter = defaultdict(list)
    majority = None
    max_count = 0

    for i in range(len(d[key])):
        if str(d[key][i][0]) not in counter:
            counter[str(d[key][i][0])] = 1
        else:
            counter[str(d[key][i][0])] += 1
        if max_count < counter[str(d[key][i][0])]:
            max_count = counter[str(d[key][i][0])]
            majority = d[key][i]
    ensemble_start[key] = majority

for key in d:
    counter = defaultdict(list)
    majority_start = None
    max_count = 0

    for i in range(len(d[key])):
        if str(d[key][i][0]) not in counter:
            counter[str(d[key][i][0])] = 1
        else:
            counter[str(d[key][i][0])] += 1
        if max_count < counter[str(d[key][i][0])]:
            max_count = counter[str(d[key][i][0])]
            majority_start = d[key][i][0]

    counter_end = {}
    majority_end = None
    max_count = 0
    for i in range(len(d[key])):
        if d[key][i][0] == majority_start:
            if str(d[key][i][1]) not in counter_end:
                counter_end[str(d[key][i][1])] = 1
            else:
                counter_end[str(d[key][i][1])] += 1
            if max_count < counter_end[str(d[key][i][1])]:
                max_count = counter_end[str(d[key][i][1])]
                majority_end = d[key][i][1]

    # if majority_end < majority_start:
    #     print("ohno4", majority_start, majority_end)
    ensemble_start_sort_end[key] = [majority_start, majority_end]

ensemble_super = {}
# voting
for key in d:
    counter = defaultdict(list)
    majority = None
    max_count = 0

    for m in [ensemble, ensemble_start_end_separate, ensemble_start, ensemble_start_sort_end]:
        start, end = m[key]
        if str(m[key]) not in counter:
            counter[str(m[key])] = 1
        else:
            counter[str(m[key])] += 1
        if max_count < counter[str(m[key])]:
            max_count = counter[str(m[key])]
            majority = m[key]
    # print(counter)
    # print("super ensemble:", max_count)
    ensemble_super[key] = majority

# Serializing json 
json_object = json.dumps(ensemble)
  
# Writing to sample.json
with open(prefix + "ensemble.json", "w") as outfile:
    outfile.write(json_object)

# # Serializing json 
# json_object = json.dumps(ensemble_start_end_separate)
  
# # Writing to sample.json
# with open(prefix + "ensemble_start_end_separate.json", "w") as outfile:
#     outfile.write(json_object)

# # Serializing json 
# json_object = json.dumps(ensemble_start)
  
# # Writing to sample.json
# with open(prefix + "ensemble_start.json", "w") as outfile:
#     outfile.write(json_object)

# # Serializing json 
# json_object = json.dumps(ensemble_start_sort_end)
  
# # Writing to sample.json
# with open(prefix + "ensemble_start_sort_end.json", "w") as outfile:
#     outfile.write(json_object)

# # Serializing json 
# json_object = json.dumps(ensemble_super)
  
# # Writing to sample.json
# with open(prefix + "ensemble_super.json", "w") as outfile:
#     outfile.write(json_object)