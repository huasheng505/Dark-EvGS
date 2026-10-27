import json
import os
from pathlib import Path
import subprocess
import sys


def test_complete_training_resume_render_evaluate(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'workflow'
    subprocess.run([sys.executable,str(root/'scripts/smoke_reproduction.py'),'--workdir',str(output)],
                   cwd=root,check=True,timeout=120,
                   env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    result = json.loads((output/'smoke_result.json').read_text())
    assert result['final_iteration']==6
    assert result['resume_matches_uninterrupted']
    assert result['test_views']==2
