import json
with open('input.json') as json_file:
    data = json.load(json_file)
data["number_of_kmc_steps"] = 500000
with open('input.json', 'w') as outfile:
    json.dump(data, outfile)
