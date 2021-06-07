import os
import os.path as op
from collections import defaultdict
import pandas
import numpy as np
from pprint import pprint
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('submissions_dir',
                    help="Directory with submitted excel spreadsheets")
parser.add_argument('output_dir',
                    help=("Output directory that the collated group marks will"
                          "be placed"))
parser.add_argument('marks', help="The file with the final marks")
args = parser.parse_args()

aspects = defaultdict(list)
peers = defaultdict(lambda: defaultdict(list))

for fname in os.listdir(args.submissions_dir):
    if fname.startswith('.') or fname.startswith('~'):
        continue
    df = pandas.read_excel(
        op.join(args.submissions_dir, fname), engine='openpyxl',
        usecols='B:G', skiprows=1, nrows=12)

    columns = iter(df.columns)

    aspects_col = df[next(columns)]

    group_name = aspects_col[11]
    aspects[group_name].append(aspects_col[:9])
    
    for name in columns:
        if name == '<NAME-OF-MEMBER>' or name == 'Unnamed: 6':
            continue
        peers[group_name][name].append(df[name][:9])

aspect_arrays = {group_name: np.array(scores, dtype=float).T
                 for group_name, scores in aspects.items()}
peer_arrays = {group_name: {member: np.array(scores, dtype=float).T
                            for member, scores in group_scores.items()}
               for group_name, group_scores in peers.items()}

aspect_weights = {}
member_scores = defaultdict(dict)
multipliers = defaultdict(dict)

for group in aspect_arrays:

    weights = np.median(aspect_arrays[group], axis=1)
    weights /= np.sum(weights)  # normalise so they add to 1.0
    aspect_weights[group] = weights * 100.0

    ngroup = len(peer_arrays[group])

    medians = {}
    totals = np.zeros(len(weights))

    for member, scores in peer_arrays[group].items():
         medians[member] = median = np.median(scores, axis=1)
         totals += median

    for member in peer_arrays[group]:
        member_scores[group][member] = ngroup * 100 * medians[member] / totals
        weighted = medians[member] * weights / totals
        multipliers[group][member] = ngroup * np.sum(weighted)

pprint(aspect_weights)
pprint(member_scores)
pprint(multipliers)
