from torch import nn

from data import NUM_LETTERS


class ShakespeareLSTM(nn.Module):
    def __init__(
        self,
        vocab_size=NUM_LETTERS,
        embedding_dim=8,
        hidden_size=256,
        num_layers=2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(self, x):
        x = self.embedding(x)

        outputs, _ = self.lstm(x)

        # LEAF uses only the final timestep.
        final = outputs[:, -1, :]

        return self.fc(final)


def create_shakespeare_lstm():
    return ShakespeareLSTM()
