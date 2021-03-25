import csv
import json

# Write a csv file with the evidence for each model and the bayes factor.

permittivities_105 = [1, 2, 4, 7, 10, 13, 16, 19, 22, 25, 28, 50, 65, 75, 85, 100]

permittivities = [1, 2, 4, 7, 10, 13, 16, 19, 22, 25, 28, 50, 75, 100]

with open('evidence_and_bayes.csv', mode='w') as csv_file:
    fieldnames = ['Number of charges', 'Relative permittivity', 'Oscillatory evidence', 'Exponential evidence', 'Bayes factor']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    # 105 charges data

    for perm in permittivities_105:
         with open("charges_105/permittivity_{}/outputs.json".format(perm)) as outfile:
             data = json.load(outfile)
         writer.writerow({'Number of charges': 105, 'Relative permittivity': perm, 'Oscillatory evidence': data["oscillatory_evidence"], 'Exponential evidence': data["dilute_evidence"], 'Bayes factor': data["bayes_factor"]})

    # 210, 421, and 2109 charges data

    for charge in [210,  421, 2109]:
        for perm in permittivities:
            with open("charges_{}/permittivity_{}/outputs.json".format(charge, perm)) as outfile:
                data = json.load(outfile)
            writer.writerow({'Number of charges': charge, 'Relative permittivity': perm, 'Oscillatory evidence': data["oscillatory_evidence"], 'Exponential evidence': data["dilute_evidence"], 'Bayes factor': data["bayes_factor"]})
