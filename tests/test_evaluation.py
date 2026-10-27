import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from evaluate import evaluate


def test_evaluation_uses_named_views_and_known_psnr(tmp_path):
    renders, truth = tmp_path/'renders',tmp_path/'gt'
    renders.mkdir()
    truth.mkdir()
    Image.fromarray(np.full((12,14,3),51,np.uint8)).save(renders/'00001.png')
    Image.fromarray(np.zeros((12,14,3),np.uint8)).save(truth/'000001.png')
    # A nonmatching GT image must not be included in the mean.
    Image.fromarray(np.full((12,14,3),255,np.uint8)).save(truth/'000002.png')
    result = evaluate(renders,truth)
    assert result['count']==1
    assert result['mean']['psnr']==pytest.approx(-10*math.log10(.2**2),abs=1e-5)
    assert result['lpips_status']=='not requested'
    assert 'lpips' not in result['mean']
    assert result['per_view'][0]['name']=='00001.png'


def test_evaluation_rejects_mismatched_resolution_and_missing_views(tmp_path):
    renders,truth = tmp_path/'renders',tmp_path/'gt'
    renders.mkdir()
    truth.mkdir()
    Image.fromarray(np.zeros((12,14,3),np.uint8)).save(renders/'000001.png')
    with pytest.raises(FileNotFoundError):
        evaluate(renders,truth)
    Image.fromarray(np.zeros((13,14,3),np.uint8)).save(truth/'000001.png')
    with pytest.raises(ValueError,match='shape mismatch'):
        evaluate(renders,truth)
