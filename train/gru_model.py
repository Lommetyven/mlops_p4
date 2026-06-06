import torch
import torch.nn as nn
import torch.nn.functional as F


class GruModel(nn.Module):
    EXPECTED_INPUT_SIZE = 16
    EXPECTED_HIDDEN_SIZE = 800
    EXPECTED_NUM_LAYERS = 1
    EXPECTED_OUTPUT_SIZE = 1
    EXPECTED_PARAMETER_BREAKDOWN = {
        "input_to_gru_weights": 38_400,
        "hidden_to_hidden_weights": 1_920_000,
        "biases": 2_400,
        "total": 1_960_800,
    }

    def __init__(
        self,
        input_size=EXPECTED_INPUT_SIZE,
        hidden_size=EXPECTED_HIDDEN_SIZE,
        num_layers=EXPECTED_NUM_LAYERS,
        output_size=EXPECTED_OUTPUT_SIZE,
    ):
        super().__init__()

        if (
            input_size != self.EXPECTED_INPUT_SIZE
            or hidden_size != self.EXPECTED_HIDDEN_SIZE
            or num_layers != self.EXPECTED_NUM_LAYERS
            or output_size != self.EXPECTED_OUTPUT_SIZE
        ):
            raise ValueError(
                "This GRU is fixed to input_size=16, hidden_size=800, "
                "num_layers=1, and output_size=1 to match 1,960,800 parameters."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.input_to_gru_weights = nn.Parameter(
            torch.empty(3 * hidden_size, input_size)
        )
        self.hidden_to_hidden_weights = nn.Parameter(
            torch.empty(3 * hidden_size, hidden_size)
        )
        self.biases = nn.Parameter(torch.empty(3 * hidden_size))

        self.reset_parameters()
        self._validate_parameter_budget()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_to_gru_weights)

        for gate_weights in self.hidden_to_hidden_weights.chunk(3, dim=0):
            nn.init.orthogonal_(gate_weights)

        nn.init.zeros_(self.biases)

    def parameter_breakdown(self):
        return {
            "input_to_gru_weights": self.input_to_gru_weights.numel(),
            "hidden_to_hidden_weights": self.hidden_to_hidden_weights.numel(),
            "biases": self.biases.numel(),
            "total": sum(parameter.numel() for parameter in self.parameters()),
        }

    def _validate_parameter_budget(self):
        actual_breakdown = self.parameter_breakdown()
        if actual_breakdown != self.EXPECTED_PARAMETER_BREAKDOWN:
            raise RuntimeError(
                "Unexpected GRU parameter count: "
                f"{actual_breakdown}. Expected {self.EXPECTED_PARAMETER_BREAKDOWN}."
            )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                "Expected input with shape (batch, sequence_length, input_size)."
            )

        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, but got {x.size(-1)}."
            )

        hidden = x.new_zeros(x.size(0), self.hidden_size)
        bias_reset, bias_update, bias_candidate = self.biases.chunk(3, dim=0)

        for x_t in x.unbind(dim=1):
            input_gates = F.linear(x_t, self.input_to_gru_weights)
            hidden_gates = F.linear(hidden, self.hidden_to_hidden_weights)

            input_reset, input_update, input_candidate = input_gates.chunk(3, dim=1)
            hidden_reset, hidden_update, hidden_candidate = hidden_gates.chunk(3, dim=1)

            reset_gate = torch.sigmoid(input_reset + hidden_reset + bias_reset)
            update_gate = torch.sigmoid(input_update + hidden_update + bias_update)
            candidate = torch.tanh(
                input_candidate + reset_gate * hidden_candidate + bias_candidate
            )

            hidden = (1 - update_gate) * candidate + update_gate * hidden

        return hidden.mean(dim=1, keepdim=True)
