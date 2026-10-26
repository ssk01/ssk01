"""Execute this project's plain Python notebook without Jupyter dependencies.

Captures stdout and matplotlib figures as standard .ipynb cell outputs.
Only use on trusted notebooks: code cells execute as local Python code.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cost-pagerank-mpl')
os.environ['MPLBACKEND'] = 'Agg'
import base64
import contextlib
import io
import json
from pathlib import Path
import traceback


def main():
    os.chdir(Path(__file__).resolve().parent)
    path = Path('cost_pagerank.ipynb')
    nb = json.loads(path.read_text())
    scope = {'__name__': '__main__'}
    counter = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        counter += 1
        outputs = []
        stream = io.StringIO()
        def show():
            import matplotlib.pyplot as plt
            for num in plt.get_fignums():
                buf = io.BytesIO()
                plt.figure(num).savefig(buf, format='png', dpi=130, bbox_inches='tight')
                outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': {
                    'image/png': base64.b64encode(buf.getvalue()).decode(),
                    'text/plain': ['<matplotlib figure>']}})
            plt.close('all')
        scope['show'] = show
        cell['execution_count'] = counter
        cell['outputs'] = outputs
        print(f'Executing cell {counter}', flush=True)
        try:
            with contextlib.redirect_stdout(stream):
                exec(compile(''.join(cell['source']), f'cell-{counter}', 'exec'), scope)
        except Exception as exc:
            outputs.append({'output_type': 'error', 'ename': type(exc).__name__,
                            'evalue': str(exc), 'traceback': traceback.format_exc().splitlines()})
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
            raise
        if stream.getvalue():
            outputs.insert(0, {'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(True)})
            print(stream.getvalue(), end='')
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f'Saved executed notebook: {path.resolve()}')


if __name__ == '__main__':
    main()
