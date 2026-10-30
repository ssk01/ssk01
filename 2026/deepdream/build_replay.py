import base64
import io
import json
from pathlib import Path
import sys
from PIL import Image

root = Path(__file__).resolve().parent
destination = Path(sys.argv[1]) if len(sys.argv)>1 else root/'replay.html'
run = json.loads((root/'inception4d/run.json').read_text())
frames = []
for frame in run['frames'][::5]:
    image = Image.open(root/frame['file'])
    image.thumbnail((600, 400))
    buffer = io.BytesIO()
    image.save(buffer, format='WEBP', quality=55)
    frames.append({'step':frame['step'],'src':'data:image/webp;base64,'+base64.b64encode(buffer.getvalue()).decode()})
html = (root/'replay-template.html').read_text().replace('PAYLOAD',json.dumps(frames))
assert len(html.encode()) < 1000000
page=(root/'page-template.html').read_text().replace('FRAGMENT_PLACEHOLDER',html)
destination.write_text(page)
print('Replay bytes:',len(html.encode()),'frames:',len(frames))
