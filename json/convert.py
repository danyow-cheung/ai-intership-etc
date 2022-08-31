import json
from os import stat
from re import L

with open("states.json") as f:
    data = json.load(f)

# data['state']大类
for state in data['states']:
    # print(state['name'],state['abbreviation'])

    del state['area_codes']

with open("new_states.json","w") as f:
    # indent --调整格式
    json.dump(data,f,indent=2)
