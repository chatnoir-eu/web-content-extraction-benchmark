import os
import subprocess
from tempfile import TemporaryDirectory


def extract(html, **_):
    cli_path = os.path.join(os.path.dirname(__file__), 'go_domdistiller_cli')

    with TemporaryDirectory() as tmp_dir:
        p = os.path.join(tmp_dir, 'go_domdistiller.html')
        with open(p, 'w') as f:
            f.write(html)
        result = subprocess.run([cli_path, p], stdout=subprocess.PIPE)
    return result.stdout.decode().strip()
