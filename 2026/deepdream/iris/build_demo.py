"""Embed the real trained model and computed trajectories into the demo."""
import argparse
import json
from pathlib import Path

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('destination',type=Path)
args=parser.parse_args()
payload=json.loads((root/'results/model.json').read_text())
keep=['W2','b2','neuron','donors','representatives','paths']
html=(root/'demo-template.html').read_text().replace('MODEL_PAYLOAD',json.dumps({k:payload[k] for k in keep}))
args.destination.write_text(html)
print('Saved demo:',args.destination.name)
