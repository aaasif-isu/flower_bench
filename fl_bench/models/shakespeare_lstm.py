"""LEAF Shakespeare next-character LSTM model definition."""

from torch import nn

ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[]"
    "abcdefghijklmnopqrstuvwxyz}"
)
NUM_LETTERS = len(ALL_LETTERS)


class ShakespeareLSTM(nn.Module):
    """2-layer character-level LSTM used in LEAF Shakespeare benchmarks."""

    def __init__(
        self,
        vocab_size: int = NUM_LETTERS,
        embedding_dim: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        outputs, _ = self.lstm(x)
        # LEAF uses only the final timestep.
        final = outputs[:, -1, :]
        return self.fc(final)


def create_shakespeare_lstm() -> nn.Module:
    return ShakespeareLSTM()
