import unittest
from lightning_ocr.metrics.recog_metric import WordMetric, CharMetric, OneMinusNEDMetric


class TestWordMetric(unittest.TestCase):
    def setUp(self):
        self.pred = [
            {"gt_text": "hello", "pred_text": "hello"},
            {"gt_text": "hello", "pred_text": "HELLO"},
            {"gt_text": "hello", "pred_text": "$HELLO$"},
        ]

    def test_word_acc_metric(self):
        metric = WordMetric(mode="exact")
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertAlmostEqual(eval_res["recog/word_acc"], 1.0 / 3, 4)

    def test_word_acc_ignore_case_metric(self):
        metric = WordMetric(mode="ignore_case")
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertAlmostEqual(eval_res["recog/word_acc_ignore_case"], 2.0 / 3, 4)

    def test_word_acc_ignore_case_symbol_metric(self):
        metric = WordMetric(mode="ignore_case_symbol")
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertEqual(eval_res["recog/word_acc_ignore_case_symbol"], 1.0)

    def test_all_metric(self):
        metric = WordMetric(mode=["exact", "ignore_case", "ignore_case_symbol"])
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=3)
        self.assertAlmostEqual(eval_res["recog/word_acc"], 1.0 / 3, 4)
        self.assertAlmostEqual(eval_res["recog/word_acc_ignore_case"], 2.0 / 3, 4)
        self.assertEqual(eval_res["recog/word_acc_ignore_case_symbol"], 1.0)


class TestCharMetric(unittest.TestCase):
    def setUp(self):
        self.pred = [
            {"gt_text": "hello", "pred_text": "helL"},
            {"gt_text": "HELLO", "pred_text": "HEL"},
        ]

    def test_char_recall_precision_metric(self):
        metric = CharMetric()
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=2)
        self.assertEqual(eval_res["recog/char_recall"], 0.7)
        self.assertEqual(eval_res["recog/char_precision"], 1)


class TestOneMinusNED(unittest.TestCase):
    def setUp(self):
        self.pred = [
            {"gt_text": "hello", "pred_text": "pred_helL"},
            {"gt_text": "HELLO", "pred_text": "HEL"},
        ]

    def test_one_minus_ned_metric(self):
        metric = OneMinusNEDMetric()
        metric.process(None, self.pred)
        eval_res = metric.evaluate(size=2)
        self.assertEqual(eval_res["recog/1-N.E.D"], 0.4875)
