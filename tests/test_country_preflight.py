"""Standard-library tests; no model, GPU, or ML packages are imported."""
from contextlib import nullcontext
from enum import Enum
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "cl" / "country_preflight.py"
SPEC = importlib.util.spec_from_file_location("country_preflight_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class ValidateBatchTests(unittest.TestCase):
    def check(self, ids=None, mask=None, labels=None, **kwargs):
        return preflight.validate_batch(
            [[1, 7, 8, 4, 2, 0]] if ids is None else ids,
            [[1, 1, 1, 1, 1, 0]] if mask is None else mask,
            [[-100, -100, -100, 4, 2, -100]] if labels is None else labels,
            kwargs.get("pad_token_id", 0), kwargs.get("eos_token_id", 2),
            kwargs.get("response_token_ids", [7, 8]))

    def test_right_padding_and_counts(self):
        report = self.check()
        self.assertEqual(report["padding_side"], "right")
        self.assertEqual(report["supervised_counts"], [2])
        self.assertEqual(report["real_lengths"], [5])
        self.assertEqual(report["response_end_positions"], [3])
        json.dumps(report, allow_nan=False)

    def test_left_padding(self):
        report = self.check([[0, 1, 7, 8, 4, 2]], [[0, 1, 1, 1, 1, 1]],
                            [[-100, -100, -100, -100, 4, 2]])
        self.assertEqual(report["padding_side"], "left")
        self.assertEqual(report["supervised_total"], 2)

    def test_unpadded(self):
        report = self.check([[1, 7, 8, 4, 2]], [[1] * 5], [[-100] * 3 + [4, 2]])
        self.assertEqual(report["padding_side"], "none")

    def test_last_response_boundary(self):
        ids = [[7, 8, 9, 2, 7, 8, 4, 2]]
        report = self.check(ids, [[1] * 8], [[-100] * 6 + [4, 2]])
        self.assertEqual(report["response_end_positions"], [6])
        with self.assertRaisesRegex(ValueError, "prompt labels"):
            self.check(ids, [[1] * 8], [[-100, -100, 9, 2, -100, -100, 4, 2]])

    def test_invalid_batches(self):
        cases = [
            ({"pad_token_id": 2}, "distinct"),
            ({"eos_token_id": None}, "integer"),
            ({"response_token_ids": []}, "nonempty"),
            ({"response_token_ids": [99]}, "boundary missing"),
            ({"response_token_ids": [8, 4, 2, 0]}, "boundary missing"),
            ({"ids": [], "mask": [], "labels": []}, "empty"),
            ({"ids": [[]], "mask": [[]], "labels": [[]]}, "empty row"),
            ({"mask": []}, "shapes"),
            ({"mask": [[1] * 5]}, "shapes"),
            ({"mask": [[0] * 6]}, "fully masked"),
            ({"mask": [[1, 1, 0, 1, 1, 0]]}, "noncontiguous"),
            ({"mask": [[0, 1, 1, 1, 1, 0]]}, "both sides"),
            ({"mask": [[1, 1, 1, 1, 1, 2]]}, "attention"),
            ({"ids": [[1, 7, 8, 4, 2, 9]]}, "padding must"),
            ({"labels": [[-100, -100, -100, 4, 2, 0]]}, "padding must"),
            ({"labels": [[1, -100, -100, 4, 2, -100]]}, "prompt labels"),
            ({"labels": [[-100, 7, 8, 4, 2, -100]]}, "prompt labels"),
            ({"labels": [[-100, -100, -100, 5, 2, -100]]}, "completion labels"),
            ({"labels": [[-100] * 6]}, "completion labels"),
            ({"labels": [[-100, -100, -100, 4, -100, -100]]}, "completion labels"),
            ({"ids": [[1, 7, 8, 4, 3, 0]],
              "labels": [[-100, -100, -100, 4, 3, -100]]}, "genuine EOS"),
            ({"ids": [[2, 7, 8, 4, 3, 0]],
              "labels": [[-100, -100, -100, 4, 3, -100]]}, "genuine EOS"),
            ({"ids": [[1, 7, 8, 0, 2, 0]],
              "labels": [[-100, -100, -100, 0, 2, -100]]}, "PAD token marked real"),
            ({"ids": [[1, 7, 8]], "mask": [[1, 1, 1]],
              "labels": [[-100] * 3]}, "no supervised"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                self.check(**kwargs)

    def test_ragged_rows(self):
        with self.assertRaisesRegex(ValueError, "shapes"):
            self.check([[7, 8, 2], [7, 8, 4, 2]], [[1] * 3, [1] * 4],
                       [[-100, -100, 2], [-100, -100, 4, 2]])

    def test_mixed_padding_sides(self):
        with self.assertRaisesRegex(ValueError, "mixed padding"):
            self.check([[7, 8, 2, 0], [0, 7, 8, 2]], [[1, 1, 1, 0], [0, 1, 1, 1]],
                       [[-100, -100, 2, -100], [-100, -100, -100, 2]])

    def test_import_without_ml_dependencies(self):
        code = (
            "import builtins, runpy; original = builtins.__import__; "
            "blocked = {'torch', 'numpy', 'unsloth', 'transformers', 'trl', 'datasets'}\n"
            "def guarded(name, *a, **kw):\n"
            "    if name.split('.')[0] in blocked: raise AssertionError(name)\n"
            "    return original(name, *a, **kw)\n"
            "builtins.__import__ = guarded\n"
            f"runpy.run_path({str(SOURCE)!r})\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "report.json"
        self.observed = []
        observed = self.observed

        class Trainer:
            def __init__(self):
                self.args = SimpleNamespace(max_steps=-1, save_strategy="epoch", warmup_steps=5)
                self.model = SimpleNamespace(revision=0)

            def train(self, *args, **kwargs):
                observed.append((self.args.max_steps, self.args.save_strategy, args, kwargs))
                self.model.revision += 1
                return SimpleNamespace(training_loss=0.25, global_step=2)

        self.original = Trainer
        self.unsloth: Any = ModuleType("unsloth")
        self.module: Any = ModuleType("unsloth.trainer")
        self.module.SFTTrainer = Trainer
        self.unsloth.trainer = self.module
        patcher = patch.dict(sys.modules, {"unsloth": self.unsloth, "unsloth.trainer": self.module})
        patcher.start()
        self.addCleanup(patcher.stop)
        audit = patch.object(preflight, "audit_trainer", return_value={"audit_passed": True})
        self.audit = audit.start()
        self.addCleanup(audit.stop)
        snapshot = patch.object(preflight, "_trainable_snapshot", side_effect=lambda model: {
            "sha256": str(model.revision), "schema_sha256": "same-schema",
            "tensor_count": 2, "parameter_count": 16, "finite": True})
        self.snapshot = snapshot.start()
        self.addCleanup(snapshot.stop)
        self.reference = SimpleNamespace(__name__="reference", run_local_unsloth_finetune=lambda: None)

    def guard(self, smoke=False):
        return preflight.guarded_reference_training(self.reference, raw_rows=[],
                                                   preflight_only=smoke, report_path=self.path)

    def report(self):
        return json.loads(self.path.read_text())

    def test_normal_training_unchanged_and_patch_restored(self):
        with self.guard():
            trainer = self.module.SFTTrainer()
            args = trainer.args
            trainer.train(trial="same-object")
            self.assertIs(trainer.args, args)
            self.assertFalse(self.path.exists())
        self.assertIs(self.module.SFTTrainer, self.original)
        self.assertEqual(self.observed, [(-1, "epoch", (), {"trial": "same-object"})])
        self.assertTrue(self.report()["passed"])
        self.assertFalse(self.report()["smoke_only"])
        self.assertEqual(self.report()["historical_runtime_parity"], "not_established")
        self.audit.assert_called_once_with(trainer, [])

    def test_smoke_only_copies_temporary_trainer_args(self):
        with self.guard(smoke=True):
            trainer = self.module.SFTTrainer()
            args = trainer.args
            trainer.train()
            self.assertIs(trainer.args, args)
            self.assertEqual(args.max_steps, -1)
            self.assertEqual(args.save_strategy, "epoch")
        self.assertEqual(self.observed, [(2, "no", (), {})])
        self.assertTrue(self.report()["smoke_only"])
        self.assertTrue(self.report()["passed"])
        self.assertEqual(self.report()["training_arguments"]["warmup_steps"], 5)
        self.assertEqual(self.report()["training_arguments"]["max_steps"], 2)
        update = self.report()["trainable_update"]
        self.assertTrue(update["changed"])
        self.assertNotEqual(update["before"]["sha256"], update["after"]["sha256"])
        self.assertEqual(update["after"]["parameter_count"], 16)
        self.assertEqual(update["historical_update_parity"], "not_established")

    def test_resume_rejected_before_audit_and_update(self):
        for positional in (True, False):
            with self.subTest(positional=positional), self.assertRaisesRegex(ValueError, "resume"):
                with self.guard():
                    trainer = self.module.SFTTrainer()
                    trainer.train("checkpoint") if positional else trainer.train(resume_from_checkpoint=True)
            self.assertFalse(self.report()["passed"])
            self.assertIs(self.module.SFTTrainer, self.original)
        self.audit.assert_not_called()
        self.assertEqual(self.observed, [])

    def test_preflight_failure_prevents_update_and_removes_stale_success(self):
        self.path.write_text('{"passed": true}')
        self.audit.side_effect = ValueError("bad masking")
        with self.assertRaisesRegex(ValueError, "bad masking"), self.guard():
            self.module.SFTTrainer().train()
        self.assertEqual(self.observed, [])
        self.assertFalse(self.report()["passed"])
        self.assertIs(self.module.SFTTrainer, self.original)

    def test_helper_failure_after_training_never_passes(self):
        with self.assertRaisesRegex(RuntimeError, "adapter save"), self.guard():
            self.module.SFTTrainer().train()
            raise RuntimeError("adapter save")
        self.assertFalse(self.report()["passed"])
        self.assertIs(self.module.SFTTrainer, self.original)

    def test_no_training_never_passes(self):
        with self.assertRaisesRegex(ValueError, "did not complete"), self.guard():
            pass
        self.assertFalse(self.report()["passed"])

    def test_repeated_train_rejected(self):
        with self.assertRaisesRegex(ValueError, "only one"), self.guard():
            trainer = self.module.SFTTrainer()
            trainer.train()
            trainer.train()
        self.assertFalse(self.report()["passed"])

    def test_nonfinite_loss_never_passes(self):
        result = SimpleNamespace(training_loss=float("nan"), global_step=2)
        with patch.object(self.original, "train", return_value=result):
            with self.assertRaisesRegex(ValueError, "nonfinite"), self.guard():
                self.module.SFTTrainer().train()
        self.assertFalse(self.report()["passed"])

    def test_wrong_smoke_steps_never_passes(self):
        result = SimpleNamespace(training_loss=0.2, global_step=1)
        with patch.object(self.original, "train", return_value=result):
            with self.assertRaisesRegex(ValueError, "exactly two"), self.guard(smoke=True):
                self.module.SFTTrainer().train()
        self.assertFalse(self.report()["passed"])

    def test_unchanged_trainables_fail_despite_finite_loss(self):
        result = SimpleNamespace(training_loss=0.2, global_step=2)
        with patch.object(self.original, "train", return_value=result):
            with self.assertRaisesRegex(ValueError, "did not change"), self.guard(smoke=True):
                self.module.SFTTrainer().train()
        self.assertFalse(self.report()["passed"])
        self.assertFalse(self.report()["trainable_update"]["changed"])

    def test_no_trainables_prevents_update(self):
        self.snapshot.side_effect = ValueError("no nonempty trainable parameters")
        with self.assertRaisesRegex(ValueError, "no nonempty"), self.guard():
            self.module.SFTTrainer().train()
        self.assertEqual(self.observed, [])
        self.assertFalse(self.report()["passed"])

    def test_nonfinite_trainables_after_training_fail(self):
        self.snapshot.side_effect = [
            {"sha256": "before", "schema_sha256": "schema", "tensor_count": 1, "parameter_count": 8},
            ValueError("nonfinite trainable parameter: adapter")]
        with self.assertRaisesRegex(ValueError, "nonfinite trainable"), self.guard():
            self.module.SFTTrainer().train()
        self.assertEqual(len(self.observed), 1)
        self.assertFalse(self.report()["passed"])

    def test_changed_trainable_schema_fails(self):
        self.snapshot.side_effect = [
            {"sha256": "before", "schema_sha256": "schema1", "tensor_count": 1, "parameter_count": 8},
            {"sha256": "after", "schema_sha256": "schema2", "tensor_count": 1, "parameter_count": 8}]
        with self.assertRaisesRegex(ValueError, "set/shape/dtype changed"), self.guard():
            self.module.SFTTrainer().train()
        self.assertFalse(self.report()["passed"])

    def test_train_exception_restores_smoke_args_and_patch(self):
        trainer = args = None
        with patch.object(self.original, "train", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"), self.guard(smoke=True):
                trainer = self.module.SFTTrainer()
                args = trainer.args
                trainer.train()
        assert trainer is not None
        self.assertIs(trainer.args, args)
        self.assertIs(self.module.SFTTrainer, self.original)
        self.assertFalse(self.report()["passed"])


class AuditTrainerTests(unittest.TestCase):
    def setUp(self):
        case = self
        self.rows = [{"prompt": str(i), "completion": "4"} for i in range(130)]
        self.ids = [[i + 10, 7, 8] + [4] * (1 + i % 3) + [151645] for i in range(130)]
        self.collated_sizes = []
        self.decoded = []
        self.bad_index = None

        class Tokenizer:
            chat_template = "exact-template"
            pad_token_id = 151669
            eos_token_id = 151645
            padding_side = "right"

            def apply_chat_template(self, messages, **kwargs):
                case.assertEqual(kwargs, {"tokenize": False, "add_generation_prompt": False,
                                          "chat_template": self.chat_template})
                return messages[0]["content"]

            def __call__(self, texts, **kwargs):
                case.assertEqual(kwargs, {"add_special_tokens": False, "truncation": False, "padding": False})
                return {"input_ids": [case.ids[int(text)] for text in texts]}

            def decode(self, ids, **kwargs):
                case.decoded.append(ids)
                return str(ids)

        def collate(features):
            self.collated_sizes.append(len(features))
            width = max(len(row["input_ids"]) for row in features)
            batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
            for row in features:
                self.assertEqual(set(row), {"input_ids", "attention_mask"})
                ids = row["input_ids"]
                padding = width - len(ids)
                labels = [-100] * 3 + ids[3:] + [-100] * padding
                if ids[0] - 10 == self.bad_index:
                    labels[0] = ids[0]
                batch["input_ids"].append(ids + [151669] * padding)
                batch["attention_mask"].append([1] * len(ids) + [0] * padding)
                batch["labels"].append(labels)
            return {key: SimpleNamespace(tolist=lambda rows=rows: rows) for key, rows in batch.items()}

        setattr(collate, "response_token_ids", [7, 8])
        self.trainer = SimpleNamespace(
            model=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="torch.bfloat16")])),
            args=SimpleNamespace(packing=False, max_seq_length=500, learning_rate=0.0002),
            processing_class=Tokenizer(), train_dataset=[{"input_ids": ids} for ids in self.ids],
            data_collator=collate)
        self.preserve_runtime = preflight._preserved_runtime
        for patcher in (patch.object(preflight, "_preserved_runtime", side_effect=lambda model: nullcontext(None)),
                        patch.object(preflight, "_logit_probe", return_value={"finite": True}),
                        patch.object(preflight.metadata, "version", return_value="mock-version")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_every_chunk_is_mask_checked_and_only_extremes_decoded(self):
        report = preflight.audit_trainer(self.trainer, self.rows)
        self.assertEqual(self.collated_sizes, [64, 64, 2, 2])
        self.assertEqual(report["rows_mask_checked"], 130)
        self.assertEqual(report["supervised_total"], sum(len(ids) - 3 for ids in self.ids))
        self.assertEqual(len(self.decoded), 2)
        self.assertEqual(report["trainer_arguments"]["max_seq_length"], 500)
        self.assertNotIn("labels inspected on selected short/long rows only", report["limits"])
        json.dumps(report, allow_nan=False)

    def test_chunk_collation_rng_is_restored_on_success_and_failure(self):
        state = {"numpy": 11, "torch": 22}
        numpy = SimpleNamespace(random=SimpleNamespace(get_state=lambda: state["numpy"],
            set_state=lambda value: state.update(numpy=value)))
        torch = SimpleNamespace(get_rng_state=lambda: state["torch"],
            set_rng_state=lambda value: state.update(torch=value),
            cuda=SimpleNamespace(is_initialized=lambda: False))
        model = self.trainer.model
        model.training = True
        model.modules = lambda: [model]
        model.train = lambda value: setattr(model, "training", value)
        original_collator = self.trainer.data_collator

        def collate(features):
            random.random()
            state["numpy"] += 1
            state["torch"] += 1
            return original_collator(features)

        setattr(collate, "response_token_ids", [7, 8])
        self.trainer.data_collator = collate
        python_state = random.getstate()
        with patch.object(preflight, "_preserved_runtime", self.preserve_runtime), \
                patch.dict(sys.modules, {"numpy": numpy, "torch": torch}):
            for bad_index in (None, 64):
                self.bad_index = bad_index
                if bad_index is None:
                    preflight.audit_trainer(self.trainer, self.rows)
                else:
                    with self.assertRaisesRegex(ValueError, "prompt labels"):
                        preflight.audit_trainer(self.trainer, self.rows)
                self.assertEqual(random.getstate(), python_state)
                self.assertEqual(state, {"numpy": 11, "torch": 22})
                self.assertTrue(model.training)

    def test_non_extreme_bad_row_in_second_chunk_fails(self):
        self.bad_index = 64
        with self.assertRaisesRegex(ValueError, "prompt labels"):
            preflight.audit_trainer(self.trainer, self.rows)
        self.assertEqual(self.collated_sizes, [64, 64])
        self.assertEqual(self.decoded, [])

    def test_qwen_runtime_ids_fail_without_repair(self):
        for name, wrong in (("pad_token_id", 0), ("eos_token_id", 2)):
            with self.subTest(name=name), patch.object(self.trainer.processing_class, name, wrong):
                with self.assertRaisesRegex(ValueError, "PAD=151669 and EOS=151645"):
                    preflight.audit_trainer(self.trainer, self.rows)
                self.assertEqual(getattr(self.trainer.processing_class, name), wrong)
        self.assertEqual(self.collated_sizes, [])

    def test_argument_whitelist_excludes_secrets_and_serializes_enum(self):
        class Optim(Enum):
            ADAMW = "adamw_torch"
        args = SimpleNamespace(optim=Optim.ADAMW, adam_beta1=0.9, adam_beta2=0.999,
            adam_epsilon=1e-8, weight_decay=0.01, learning_rate=0.0002, lr_scheduler_type="linear",
            warmup_steps=5, warmup_ratio=0.0, per_device_train_batch_size=22,
            gradient_accumulation_steps=3, seed=1, data_seed=None, max_grad_norm=1.0,
            fp16=False, bf16=True, packing=False, max_seq_length=500, hub_token="SECRET")
        report = preflight._trainer_arguments(args)
        self.assertEqual(report["optim"], "adamw_torch")
        for name, value in vars(args).items():
            if name not in ("optim", "hub_token"):
                self.assertEqual(report[name], value)
        self.assertNotIn("SECRET", json.dumps(report))
        self.assertNotIn("hub_token", report)


class TrainableSnapshotTests(unittest.TestCase):
    def setUp(self):
        class Tensor:
            dtype = "torch.float32"
            requires_grad = True

            def __init__(self, values):
                self.values = values
                self.shape = (len(values),)

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def reshape(self, *shape):
                return self

            def view(self, dtype):
                return self

            def numpy(self):
                return SimpleNamespace(tobytes=lambda: struct.pack(f"<{len(self.values)}f", *self.values))

            def numel(self):
                return len(self.values)

        self.Tensor = Tensor
        torch = SimpleNamespace(uint8="uint8", isfinite=lambda tensor: SimpleNamespace(
            all=lambda: all(math.isfinite(value) for value in tensor.values)))
        patcher = patch.dict(sys.modules, {"torch": torch})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_hashes_every_trainable_and_ignores_frozen_values(self):
        first, second, frozen = self.Tensor([1.0]), self.Tensor([2.0, 3.0]), self.Tensor([4.0])
        frozen.requires_grad = False
        model = SimpleNamespace(named_parameters=lambda: iter([("b", second), ("a", first), ("base", frozen)]))
        state = random.getstate()
        before = preflight._trainable_snapshot(model)
        self.assertEqual(before["tensor_count"], 2)
        self.assertEqual(before["parameter_count"], 3)
        frozen.values[0] = float("nan")
        self.assertEqual(before, preflight._trainable_snapshot(model))
        second.values[1] = 3.125
        after = preflight._trainable_snapshot(model)
        self.assertNotEqual(before["sha256"], after["sha256"])
        self.assertEqual(before["schema_sha256"], after["schema_sha256"])
        self.assertEqual(random.getstate(), state)

    def test_no_trainables_or_nonfinite_trainables_fail(self):
        for values, message in (([], "no nonempty"), ([float("nan")], "nonfinite"),
                                ([float("inf")], "nonfinite")):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                preflight._trainable_snapshot(SimpleNamespace(
                    named_parameters=lambda: iter([("adapter", self.Tensor(values))])))
        with self.assertRaisesRegex(ValueError, "no nonempty"):
            preflight._trainable_snapshot(SimpleNamespace(named_parameters=lambda: iter([])))


class RuntimeStateTests(unittest.TestCase):
    def test_rng_and_model_modes_restored_even_on_failure(self):
        state = {"np": 11, "torch": 22, "cuda": [33]}
        numpy: Any = ModuleType("numpy")
        numpy.random = SimpleNamespace(get_state=lambda: state["np"],
                                       set_state=lambda value: state.update(np=value))
        torch: Any = ModuleType("torch")
        torch.get_rng_state = lambda: state["torch"]
        torch.set_rng_state = lambda value: state.update(torch=value)
        torch.cuda = SimpleNamespace(is_initialized=lambda: True,
            get_rng_state_all=lambda: state["cuda"],
            set_rng_state_all=lambda value: state.update(cuda=value))
        child = SimpleNamespace(training=False)

        class Model:
            training = True

            def modules(self):
                return [self, child]

            def train(self, mode):
                self.training = child.training = mode

        model = Model()
        python_state = random.getstate()
        with patch.dict(sys.modules, {"numpy": numpy, "torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "probe"), preflight._preserved_runtime(model):
                random.random()
                state.update(np=99, torch=88, cuda=[77])
                model.train(False)
                raise RuntimeError("probe")
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(state, {"np": 11, "torch": 22, "cuda": [33]})
        self.assertTrue(model.training)
        self.assertFalse(child.training)


if __name__ == "__main__":
    unittest.main()
