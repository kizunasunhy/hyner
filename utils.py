# coding=utf-8
# Copyright 2019 SK T-Brain Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import requests
import hashlib
import random
import torch
import json
import numpy as np
from pathlib import Path

kobert_models = {
    'onnx_kobert': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/onnx/onnx_kobert_44529811f0.onnx',
        'fname': 'onnx_kobert_44529811f0.onnx',
        'chksum': '44529811f0'
    },
    'tokenizer': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/tokenizer/tokenizer_78b3253a26.model',
        'fname': 'tokenizer_78b3253a26.model',
        'chksum': '78b3253a26'
    }
}


def download(url, filename, chksum, cachedir='./'):
    f_cachedir = os.path.expanduser(cachedir)
    os.makedirs(f_cachedir, exist_ok=True)
    file_path = os.path.join(f_cachedir, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path,
                            'rb').read()).hexdigest()[:10] == chksum:
            print('using cached model')
            return file_path
    with open(file_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                    chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done,
                                                   '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
    assert chksum == hashlib.md5(open(
        file_path, 'rb').read()).hexdigest()[:10], 'corrupted file!'
    return file_path


def get_onnx(cachedir='./'):
    """Get KoBERT ONNX file path after downloading
    """
    model_info = kobert_models['onnx_kobert']
    return download(model_info['url'],
                    model_info['fname'],
                    model_info['chksum'],
                    cachedir=cachedir)

def get_tokenizer(cachedir='./'):
    """Get KoBERT Tokenizer file path after downloading
    """
    model_info = kobert_models['tokenizer']
    return download(model_info['url'],
                        model_info['fname'],
                        model_info['chksum'],
                        cachedir=cachedir)  

def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        
        
class CheckpointManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir

    def save_checkpoint(self, state, filename):
        torch.save(state, self._model_dir / filename)

    def load_checkpoint(self, filename):
        state = torch.load(self._model_dir / filename, map_location=torch.device('cpu'))
        return state


class SummaryManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir
        self._summary = {}

    def save(self, filename):
        with open(self._model_dir / filename, mode='w') as io:
            json.dump(self._summary, io, indent=4)

    def load(self, filename):
        with open(self._model_dir / filename, mode='r') as io:
            metric = json.loads(io.read())
        self.update(metric)

    def update(self, summary):
        self._summary.update(summary)

    def reset(self):
        self._summary = {}

    @property
    def summary(self):
        return self._summary