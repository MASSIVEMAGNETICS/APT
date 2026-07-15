from pathlib import Path
from tempfile import TemporaryDirectory
import math
import unittest

from apt.models import APTByteModel, RNNByteModel, TransformerByteModel


CORPUS = (b"branch memory learns patterns. " * 24) + (b"state memory predicts bytes. " * 24)


class ByteModelTests(unittest.TestCase):
    def test_apt_training_reduces_training_loss(self) -> None:
        model = APTByteModel(decays=(0.0, 0.75), seed=7)
        before = model.loss(CORPUS)
        history = model.train(CORPUS, epochs=2, learning_rate=0.02, batch_tokens=64)
        after = model.loss(CORPUS)
        self.assertTrue(all(math.isfinite(value) for value in history))
        self.assertLess(after, before)

    def test_models_generate_deterministically_and_round_trip(self) -> None:
        models = [
            APTByteModel(decays=(0.0,), seed=3),
            RNNByteModel(hidden_size=12, seed=3),
            TransformerByteModel(model_size=8, context_length=8, feedforward_size=12, seed=3),
        ]
        with TemporaryDirectory() as directory:
            for index, model in enumerate(models):
                path = Path(directory) / f"model-{index}.npz"
                generated = model.generate(b"APT", max_new_bytes=8, temperature=0.7, seed=11)
                self.assertEqual(generated, model.generate(b"APT", max_new_bytes=8, temperature=0.7, seed=11))
                self.assertEqual(len(generated), 11)
                model.save(path)
                restored = type(model).load(path)
                self.assertAlmostEqual(model.loss(CORPUS[:128]), restored.loss(CORPUS[:128]), places=12)

    def test_rnn_and_transformer_training_are_finite(self) -> None:
        data = CORPUS[:256]
        rnn = RNNByteModel(hidden_size=12, seed=5)
        transformer = TransformerByteModel(
            model_size=8, context_length=8, feedforward_size=12, seed=5
        )
        self.assertTrue(math.isfinite(rnn.train(data, epochs=1, learning_rate=0.003)[0]))
        self.assertTrue(math.isfinite(transformer.train(data, epochs=1, learning_rate=0.001)[0]))
        self.assertTrue(math.isfinite(rnn.loss(data)))
        self.assertTrue(math.isfinite(transformer.loss(data)))


if __name__ == "__main__":
    unittest.main()

