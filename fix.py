import json
from collections import defaultdict


def myhook(pairs):
    d = {}
    for k, v in pairs:
        if k not in d:
          d[k] = v
        else:
          d[k] += v
    return d

mydata = json.loads('users-new copy.json', object_pairs_hook=myhook)

with open("users-new copy.json", "w", encoding="utf-8") as convert_file:
    convert_file.write((json.dumps(mydata, indent=4)))
