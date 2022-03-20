import json
import argparse
import pprint
import csv

parser = argparse.ArgumentParser(description="parse squad qa into scv")
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()
input_file = args.input
output_file = args.output

with open(input_file, 'r') as f:
    data = json.load(f)

pprint.pprint(data['data'][0]['paragraphs'][0]['qas'])
print(len(data['data']))

i = 0
output_data = []
for d in data['data']:
    for paragraph in d['paragraphs']:
        for qa in paragraph['qas']:
            if qa['answers']:
                print(qa['question'])
                print(qa['answers'][0]['text'])
                output_data.append([str(i), qa['question'], qa['answers'][0]['text']])
                i += 1

with open(output_file, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(['id', 'question', 'answer'])

    # write multiple rows
    writer.writerows(output_data)
