import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import mask_padding_labels, split_records
from scripts.dpo_core import dpo_loss, preference_accuracy
from scripts.eval_models import extract_label
from scripts.grpo_core import clipped_grpo_loss, compute_group_advantages
from scripts.grpo_smoke import reward_response
from scripts.api_server import ADAPTER_PATH
from scripts.train_domain import INITIAL_ADAPTER_PATH
from scripts.train_dpo import POLICY_PATH, REFERENCE_PATH
from scripts.ablation import ABLATION_CONFIGS, select_best_validation


class DataUtilityTests(unittest.TestCase):
    def test_split_records_is_deterministic_and_non_empty(self):
        records = list(range(10))
        first = split_records(records, val_ratio=0.2, seed=17)
        second = split_records(records, val_ratio=0.2, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(set(first[0]) | set(first[1]), set(records))
        self.assertTrue(set(first[0]).isdisjoint(first[1]))
        self.assertEqual(len(first[1]), 2)

    def test_padding_tokens_are_ignored_by_supervised_loss(self):
        input_ids = torch.tensor([[4, 5, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])

        labels = mask_padding_labels(input_ids, attention_mask)

        self.assertEqual(labels.tolist(), [[4, 5, -100, -100]])

    def test_checked_in_domain_data_is_valid_json(self):
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "data" / "domain_zh.json").read_text(encoding="utf-8"))
        preferences = json.loads(
            (root / "data" / "dpo_domain_demo.json").read_text(encoding="utf-8")
        )
        evaluation = json.loads(
            (root / "data" / "eval_domain.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(data), 30)
        self.assertGreaterEqual(len(preferences), 8)
        self.assertTrue(all({"instruction", "input", "output"} <= item.keys() for item in data))
        self.assertTrue(all({"instruction", "chosen", "rejected"} <= item.keys() for item in preferences))
        self.assertTrue(all({"task", "instruction", "input", "label", "labels"} <= item.keys() for item in evaluation))


class PreferenceOptimizationTests(unittest.TestCase):
    def test_dpo_loss_decreases_for_a_stronger_chosen_margin(self):
        weak = dpo_loss(
            torch.tensor([1.0]), torch.tensor([0.0]),
            torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1
        )
        strong = dpo_loss(
            torch.tensor([4.0]), torch.tensor([0.0]),
            torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1
        )
        self.assertLess(strong.item(), weak.item())

    def test_preference_accuracy_uses_reference_adjusted_margin(self):
        accuracy = preference_accuracy(
            torch.tensor([2.0, 0.0]), torch.tensor([0.0, 3.0]),
            torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])
        )
        self.assertEqual(accuracy, 0.5)


class GrpoTests(unittest.TestCase):
    def test_group_advantage_is_normalized_within_each_prompt(self):
        rewards = torch.tensor([[1.0], [3.0], [10.0], [10.0]])
        response_mask = torch.ones_like(rewards)
        group_ids = ["a", "a", "b", "b"]

        advantages = compute_group_advantages(rewards, response_mask, group_ids)

        self.assertAlmostEqual(advantages[0, 0].item(), -1.0, places=5)
        self.assertAlmostEqual(advantages[1, 0].item(), 1.0, places=5)
        self.assertEqual(advantages[2, 0].item(), 0.0)
        self.assertEqual(advantages[3, 0].item(), 0.0)

    def test_masked_tokens_have_zero_advantage(self):
        rewards = torch.tensor([[1.0, 99.0], [3.0, 99.0]])
        response_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        advantages = compute_group_advantages(rewards, response_mask, [0, 0])

        self.assertEqual(advantages[:, 1].tolist(), [0.0, 0.0])

    def test_clipped_grpo_loss_prefers_positive_advantage_updates(self):
        loss = clipped_grpo_loss(
            torch.tensor([1.0, 1.5]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 1.0]),
            clip_eps=0.2,
        )
        self.assertLess(loss.item(), 0.0)

    def test_reward_function_requires_the_expected_answer(self):
        self.assertEqual(reward_response("答案：4", "4"), 1.0)
        self.assertEqual(reward_response("答案：5", "4"), 0.0)
        self.assertEqual(reward_response("答案：14", "4"), 0.0)


class EvaluationTests(unittest.TestCase):
    def test_extract_label_matches_only_the_declared_label_set(self):
        self.assertEqual(extract_label("答案：科技区-编程", ["科技区-编程", "生活区-美食"]), "科技区-编程")
        self.assertEqual(extract_label("无法判断", ["技术", "生活"]), "<unparsed>")

    def test_api_serves_the_final_dpo_adapter(self):
        self.assertEqual(ADAPTER_PATH.name, "domain_dpo")

    def test_training_stages_form_a_serial_pipeline(self):
        self.assertEqual(INITIAL_ADAPTER_PATH.name, "general_sft")
        self.assertEqual(POLICY_PATH.name, "domain_sft")
        self.assertEqual(REFERENCE_PATH.name, "domain_sft")

    def test_ablation_configs_have_unique_names_and_select_best_validation(self):
        names = [config["name"] for config in ABLATION_CONFIGS]
        self.assertEqual(len(names), len(set(names)))
        best = select_best_validation({
            "short": {"validation": [3.0, 2.0]},
            "long": {"validation": [3.0, 1.5]},
        })
        self.assertEqual(best, "long")


if __name__ == "__main__":
    unittest.main()
