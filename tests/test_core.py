import tempfile
import unittest
from pathlib import Path

import pandas as pd

from auto_classifier.data import DataFormatError, load_tables, normalize_human_answer
from auto_classifier.model import HybridValidator, choose_low_threshold, evaluate_low_threshold
from auto_classifier.config import ValidatorConfig
from auto_classifier.prepare import normalize_messages_table, prepare_training_data
from auto_classifier.subreason_mapping import (
    apply_subreason_mapping,
    load_subreason_mapping,
    use_subreason_key_as_reason_id,
)
from auto_classifier.text import split_roles


class DataTests(unittest.TestCase):
    def test_answer_normalization(self):
        self.assertEqual(normalize_human_answer("да"), 1)
        self.assertEqual(normalize_human_answer("нет"), 0)
        self.assertEqual(normalize_human_answer("0"), 0)
        self.assertEqual(normalize_human_answer("да?"), 1)
        self.assertEqual(normalize_human_answer("нет?"), 0)
        self.assertIsNone(normalize_human_answer(""))

    def test_role_split(self):
        parsed = split_roles("client: хочу продлить\nmanager: оформим\nbot: ждите")
        self.assertEqual(parsed.client_text, "хочу продлить")
        self.assertEqual(parsed.operator_text, "оформим")
        self.assertEqual(parsed.bot_text, "ждите")
        self.assertEqual(parsed.model_text, "хочу продлить")

    def test_missing_chat_text_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            pd.DataFrame(
                {"chat_id": ["1"], "reason_id": ["1"], "да/нет": ["да"]}
            ).to_csv(path, index=False)
            with self.assertRaises(DataFormatError):
                load_tables([str(path)], require_text=True, require_answer=True)

    def test_mixed_alias_columns_are_coalesced_per_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.csv"
            pd.DataFrame(
                {
                    "comm_id": ["T1", ""],
                    "chat_id": ["", "T2"],
                    "reason_number": ["1", ""],
                    "reason_id": ["", "2"],
                    "да/нет": ["да", "нет"],
                    "chat_text": ["client: one", "client: two"],
                }
            ).to_csv(path, index=False)
            frame = load_tables([str(path)], require_text=True, require_answer=True)
            self.assertEqual(frame["chat_id"].tolist(), ["T1", "T2"])
            self.assertEqual(frame["reason_id"].tolist(), ["1", "2"])
            self.assertEqual(frame["human_label"].tolist(), [1, 0])

    def test_prepare_messages_and_labels(self):
        messages = pd.DataFrame(
            {
                "ID_diologa": ["T1", "T1", "T2"],
                "Vremya": [
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:01:00",
                    "2026-01-02 11:00:00",
                ],
                "Kto": ["client", "manager", "bot"],
                "Soobschenie": ["хочу продлить", "поможем", "уточните"],
            }
        )
        chats = normalize_messages_table(messages)
        self.assertEqual(len(chats), 2)
        chat_text = chats[chats["chat_id"] == "T1"]["chat_text"].iloc[0]
        self.assertIn("client: хочу продлить", chat_text)
        self.assertIn("manager: поможем", chat_text)

        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.csv"
            messages_path = Path(tmp) / "messages.csv"
            out_path = Path(tmp) / "prepared.csv"
            pd.DataFrame(
                {
                    "comm_id": ["T1"],
                    "reason_numb": ["2"],
                    "да/нет": ["да"],
                    "комментарий": ["ok"],
                }
            ).to_csv(labels_path, index=False)
            messages.to_csv(messages_path, index=False)
            prepared, stats = prepare_training_data(
                labels_paths=[str(labels_path)],
                messages_paths=[str(messages_path)],
                output=str(out_path),
            )
            self.assertTrue(out_path.exists())
            self.assertEqual(stats.matched_rows, 1)
            self.assertEqual(prepared["reason_id"].iloc[0], "2")
            self.assertTrue(prepared["has_chat_text"].iloc[0])

    def test_prepare_full_dialog_export(self):
        messages = pd.DataFrame(
            {
                "ID_diologa": ["T1"],
                "Kolichestvo_soobscheniy": ["3"],
                "Pervoe_soobschenie": ["2026-01-01 10:00:00"],
                "Poslednee_soobschenie": ["2026-01-01 10:02:00"],
                "Dialog_polnostyu": [
                    "2026-01-01 10:00:00 | CLIENT | хочу продлить\n"
                    "2026-01-01 10:01:00 | BOT | уточните\n"
                    "2026-01-01 10:02:00 | MANAGER | поможем"
                ],
            }
        )
        chats = normalize_messages_table(messages)
        self.assertEqual(len(chats), 1)
        self.assertEqual(chats["message_count"].iloc[0], 3)
        self.assertEqual(chats["client_message_count"].iloc[0], 1)
        self.assertEqual(chats["manager_message_count"].iloc[0], 1)
        self.assertEqual(chats["bot_message_count"].iloc[0], 1)
        self.assertIn("client: хочу продлить", chats["chat_text"].iloc[0])
        self.assertIn("manager: поможем", chats["chat_text"].iloc[0])

    def test_subreason_mapping_merges_old_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "map.yaml"
            mapping_path.write_text(
                """
datasets:
  kasko_uregulirovanie:
    files:
      - labels.xlsx
    iterations:
      "итерация 1":
        reasons:
          "1": unclear_status_and_communication
          "4": unclear_status_and_communication
          "7": unclear_status_and_communication
          "2": service_selection_change
""",
                encoding="utf-8",
            )
            mapping = load_subreason_mapping(str(mapping_path))
            frame = pd.DataFrame(
                {
                    "chat_id": ["T1", "T2", "T3"],
                    "reason_id": ["1", "4", "2"],
                    "_source_file": ["/tmp/labels.xlsx"] * 3,
                    "_source_sheet": ["итерация 1"] * 3,
                }
            )
            mapped = apply_subreason_mapping(frame, mapping)
            self.assertEqual(
                mapped["subreason_key"].tolist(),
                [
                    "unclear_status_and_communication",
                    "unclear_status_and_communication",
                    "service_selection_change",
                ],
            )
            grouped = use_subreason_key_as_reason_id(mapped)
            self.assertEqual(grouped["reason_id"].iloc[0], "unclear_status_and_communication")
            self.assertEqual(grouped["original_reason_id"].iloc[0], "1")


class ModelTests(unittest.TestCase):
    def _dataset(self):
        rows = []
        positives = [
            "client: хочу продлить полис каско\nmanager: сейчас рассчитаем",
            "client: продлите страховку каско на следующий год",
            "client: нужна пролонгация каско",
            "client: хочу оформить продление полиса",
            "client: можно продлить каско?",
            "client: продление страховки интересует",
        ]
        negatives = [
            "client: дорого у вас, сравню с другой страховой",
            "client: что такое франшиза?",
            "manager: можно продлить каско?\nclient: мне нужен осаго",
            "client: хочу расторгнуть полис",
            "client: ошибка в приложении",
            "client: оформил в другой страховой",
        ]
        for i, text in enumerate(positives):
            rows.append({"chat_id": f"p{i}", "chat_text": text, "reason_id": "1", "да/нет": "да"})
        for i, text in enumerate(negatives):
            rows.append({"chat_id": f"n{i}", "chat_text": text, "reason_id": "1", "да/нет": "нет"})
        rows.append({"chat_id": "low1", "chat_text": "client: отказали", "reason_id": "8", "да/нет": "да"})
        rows.append({"chat_id": "low2", "chat_text": "client: отказа нет", "reason_id": "8", "да/нет": "нет"})
        return pd.DataFrame(rows)

    def test_choose_low_threshold_uses_separate_precision_target(self):
        threshold, precision, coverage = choose_low_threshold(
            y=[1, 1, 0, 0],
            probabilities=[0.9, 0.8, 0.2, 0.1],
            target_precision=1.0,
        )
        self.assertEqual(threshold, 0.2)
        self.assertEqual(precision, 1.0)
        self.assertEqual(coverage, 0.5)

        threshold, precision, coverage = choose_low_threshold(
            y=[1, 1, 0, 0],
            probabilities=[0.9, 0.8, 0.2, 0.1],
            target_precision=1.01,
        )
        self.assertLess(threshold, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(coverage, 0.0)

    def test_evaluate_low_threshold_cap(self):
        threshold, precision, coverage = evaluate_low_threshold(
            y=[1, 0, 0, 1],
            probabilities=[0.05, 0.03, 0.04, 0.20],
            threshold=0.04,
        )
        self.assertEqual(threshold, 0.04)
        self.assertEqual(precision, 1.0)
        self.assertEqual(coverage, 0.5)

    def test_train_predict_and_low_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            self._dataset().to_csv(path, index=False)
            frame = load_tables([str(path)], require_text=True, require_answer=True)
            config = ValidatorConfig(
                target_precision=0.8,
                use_embeddings=False,
                min_reason_samples=4,
                min_class_samples=2,
            )
            model = HybridValidator.train(frame, config)
            out_dir = Path(tmp) / "model"
            model.save(str(out_dir))
            loaded = HybridValidator.load(str(out_dir))
            predictions = loaded.predict(frame)
            self.assertIn("decision", predictions.columns)
            self.assertIn("auto_answer", predictions.columns)
            self.assertIn("p_correct", predictions.columns)
            self.assertIn("yes_threshold", predictions.columns)
            self.assertIn("no_threshold", predictions.columns)
            self.assertTrue(set(predictions["decision"]).issubset({"auto_yes", "auto_no", "review"}))
            self.assertTrue(set(predictions["auto_answer"]).issubset({"да", "нет", "review"}))
            self.assertNotIn("auto_no", set(predictions["decision"]))
            self.assertTrue((predictions["no_threshold"] < 0).all())
            low_data = loaded.reason_validators["8"]
            self.assertGreater(low_data.threshold, 1.0)
            self.assertLess(low_data.no_threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
