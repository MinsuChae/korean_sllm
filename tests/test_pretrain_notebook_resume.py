"""Run the repository's pretraining trainer on a small CPU model; no Colab required."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch._dynamo  # Load lazy optimizer imports before patch.dict restores sys.modules.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        class Tokenizer:
            def serialized_model_proto(self): return b'test-tokenizer'
            def get_piece_size(self): return 128
        data = types.ModuleType('data')
        data.load_tokenizer = Tokenizer
        pretrain = types.ModuleType('pretrain_data')
        self.window_count = 16
        self.seen = []
        owner = self
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, split):
                self.split = split
                path = owner.root / (split + '.bin')
                if not path.exists():
                    np.arange(32, dtype=np.uint16).tofile(path)
                self.tokens = np.memmap(path, dtype=np.uint16, mode='r')
            def __len__(self): return owner.window_count if self.split == 'train' else 2
            def __getitem__(self, index):
                if self.split == 'train': owner.seen.append(index)
                ids = (torch.arange(8) + index) % 128
                return {'input_ids': ids, 'loss_mask': torch.ones_like(ids)}
        pretrain.make_pretrain_dataset = lambda root, split, *args: Dataset(split)
        self.modules = {'data': data, 'pretrain_data': pretrain}
        self.trainer = types.ModuleType('notebook_trainer')
        self.trainer.__file__ = str(ROOT / 'train_pretrain.py')
        with patch.dict(sys.modules, self.modules):
            exec(compile((ROOT / 'train_pretrain.py').read_text(), str(ROOT / 'train_pretrain.py'), 'exec'), self.trainer.__dict__)
        self.trainer.PRESETS['tiny'].update(n_layers=1, d_model=16, n_heads=2,
            n_kv_heads=1, head_dim=8, ffn_hidden=32, mtp_ffn_hidden=16,
            vocab_size=128, mtp_n=2)

    def run_session(self, folder, session=0.25, epochs=5, resume=True, extra=()):
        ckpt_dir = self.root / folder
        args = ['train.py', '--preset', 'tiny', '--seq-len', '8', '--overlap', '2',
                '--batch-size', '2', '--grad-accum', '2', '--epochs', str(epochs),
                '--session-epochs', str(session), '--num-workers', '0', '--warmup-steps', '2',
                '--eval-every', '1000', '--save-every', '1000', '--eval-batches', '1',
                '--ckpt-dir', str(ckpt_dir), *extra]
        if resume and (ckpt_dir / 'last.pt').exists():
            args += ['--resume', str(ckpt_dir / 'last.pt')]
        with patch.dict(sys.modules, self.modules), patch.object(sys, 'argv', args), \
                patch('torch.cuda.is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            self.trainer.main()
        return torch.load(ckpt_dir / 'last.pt', weights_only=True)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left: self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_twenty_sessions_match_continuous_five_epochs(self):
        full = self.run_session('full', session=5)
        expected_order = self.seen.copy()
        self.seen.clear()
        for i in range(20):
            split = self.run_session('split')
            self.assertEqual(split['step'], i + 1)
        self.assertEqual(self.seen, expected_order)
        for key in ('model', 'optim', 'rng_state', 'run_config', 'data_epoch', 'batch_in_epoch'):
            self.assert_state_equal(full[key], split[key])
        self.assertEqual(split['data_epoch'], 5)
        self.assertEqual(split['batch_in_epoch'], 0)
        before = len(self.seen)
        self.run_session('split')
        self.assertEqual(len(self.seen), before)

    def test_tail_batch_and_partial_accumulation_cover_each_epoch(self):
        self.window_count = 17
        full = self.run_session('full', session=5)
        order = self.seen.copy()
        for i in range(5):
            self.assertEqual(sorted(order[i * 17:(i + 1) * 17]), list(range(17)))
        self.seen.clear()
        for _ in range(13):
            split = self.run_session('split')
        self.assertEqual(split['step'], 25)
        self.assertEqual(self.seen, order)
        self.assert_state_equal(full['model'], split['model'])
        self.assert_state_equal(full['optim'], split['optim'])

    def test_reject_changed_schedule_and_legacy_checkpoint(self):
        state = self.run_session('split')
        with self.assertRaisesRegex(ValueError, '재개 설정/데이터'):
            self.run_session('split', epochs=6)
        del state['run_config']
        torch.save(state, self.root / 'split' / 'last.pt')
        with self.assertRaisesRegex(ValueError, '이전 형식'):
            self.run_session('split')


if __name__ == '__main__':
    unittest.main()
