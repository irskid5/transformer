import torch
import math


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled Dot-Product attention module from "Attention is all you need".
    """

    def __init__(self, scale: float | None = None):
        """Initializes the ScaledDotProductAttention module, build according to "Attention is all you need".

        Args:
            scale (float | None, optional): scale to use instead of default scale. Defaults to None.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward step of ScaledDotProductAttention from "Attention is all you need".

        Args:
            q (torch.Tensor): query tensor
            k (torch.Tensor): key tensor
            v (torch.Tensor): value tensor
            mask (torch.Tensor | None, optional): mask tensor for autoregressive behaviour. Defaults to None.

        Raises:
            RuntimeError: if query and key shapes are not identical.
            RuntimeError: if sequence dimension differs across queries, keys, and values. 
            RuntimeError: if mask does not have shape of [sequence length]

        Returns:
            torch.Tensor: 
        """
        # Check dimensions
        if (q.shape != k.shape):
            raise RuntimeError("Query and Key shapes should be identical.")
        if (q.shape[-2] != v.shape[-2]):
            raise RuntimeError(
                "Query, Key, and Value shapes should have the same sequence dimension.")
        if (mask != None and (mask.shape[-1] != [q.shape[-2]])):
            raise RuntimeError(
                "Query, Key, Value, and mask shapes should have the same sequence dimension.")

        # Dot-product (similarity measure, q_dim = [bs, seq, q_k], k_dim = [bs, seq, q_k], out = [bs, seq, seq])
        matmul_0 = torch.matmul(q, k.transpose(-2, -1))

        # Scaling
        scale = 1.0 / math.sqrt(q.shape[-1]
                                ) if self.scale == None else self.scale
        scaled = scale * matmul_0

        # Masking
        masked = scaled
        if (mask != None):
            masked = scaled * mask

        # Softmax (relative scores)
        scores = torch.softmax(masked, -1)

        # Final dot-product (weighting by scores, scores_dim = [bs, seq, seq], v_dim = [bs, seq, d_v], out = [bs, seq, d_v])
        matmul_1 = torch.matmul(scores, torch.transpose(v, -2, -1))

        return matmul_1


class MultiHeadedAttention(torch.nn.Module):
    """
    Creates a multi-headed attention module from "Attention is all you need".
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, h: int = 1, scale: float | None = None, device=torch.device("cpu")):
        """Initializes the multi-headed attention module, build according to "Attention is all you need"

        Args:
            d_model (int): dimension of model
            d_k (int): dimension of queries and keys
            d_v (int): dimension of values
            h (int, optional): number of heads. Defaults to 1.
            scale (float | None, optional): scale to use instead of default. Defaults to None.
            device (_type_, optional): device to use for computation. Defaults to torch.device("cpu").
        """
        super(MultiHeadedAttention, self).__init__()
        self.device = device
        self.scale = scale
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.query_linear = torch.nn.ModuleList([torch.nn.Linear(
            d_model, d_k, bias=False, device=self.device) for _ in range(h)])
        self.key_linear = torch.nn.ModuleList([torch.nn.Linear(
            d_model, d_k, bias=False, device=self.device) for _ in range(h)])
        self.value_linear = torch.nn.ModuleList([torch.nn.Linear(
            d_model, d_v, bias=False, device=self.device) for _ in range(h)])
        self.attention_layers = torch.nn.ModuleList([
            ScaledDotProductAttention(scale) for _ in range(h)])
        self.output_linear = torch.nn.Linear(
            h*d_v, d_model, bias=False, device=self.device)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward step for MultiHeadedAttention from "Attention is all you need". Implementation
        does not run heads in parallel.

        Args:
            q (torch.Tensor): query tensor
            k (torch.Tensor): key tensor
            v (torch.Tensor): value tensor
            mask (torch.Tensor | None, optional): mask tensor for autoregressive behaviour. Defaults to None.

        Returns:
            torch.Tensor: output of forward step
        """
        attended_values = []
        for i in range(self.h):
            # Linear layers
            queries = self.query_linear[i](q)
            keys = self.key_linear[i](k)
            values = self.value_linear[i](v)

            # Attention
            attention_output = self.attention_layers[i](
                queries, keys, values, mask)

            # Add to list of heads
            attended_values.append(attention_output)

        # Output linear
        concatenated = torch.concatenate(attended_values, -1)
        output_linear = self.output_linear(concatenated)

        return output_linear


class TransformerDecoderModule(torch.nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, d_ff: int, h: int = 1, scale: float | None = None, dropout_p: float = 0.0, activation=torch.relu, device=torch.device("cpu")):
        """Initializes the transformer decoder module, build according to "Attention is all you need".
        The transformer decoder architecture stacks multiple of these decoder modules.

        Args:
            d_model (int): dimension of model
            d_k (int): dimension of queries and keys
            d_v (int): dimension of values
            d_ff (int): intermediate dimension of feedforward sublayer
            h (int, optional): number of heads. Defaults to 1.
            scale (float | None, optional): scale to use instead of default. Defaults to None.
            dropout_p (float, optional): probability for dropout in sublayers. Defaults to 0.0.
            activation ((input: torch.Tensor) -> torch.Tensor, optional): activation function for feedforward sublayer. Defaults to torch.relu.
            device (_type_, optional): device to use for computation. Defaults to torch.device("cpu").
        """
        super(TransformerDecoderModule, self).__init__()
        self.device = device
        self.scale = scale
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.activation = activation
        self.dropout_p = dropout_p

        self.multi_headed_attention = MultiHeadedAttention(
            d_model, d_k, d_v, h, scale, device)

        self.feed_forward_0 = torch.nn.Linear(d_model, d_ff)
        self.feed_forward_1 = torch.nn.Linear(d_ff, d_model)

        self.layer_norm_0 = torch.nn.LayerNorm(d_model)
        self.layer_norm_1 = torch.nn.LayerNorm(d_model)

        self.dropout_0 = torch.nn.Dropout(dropout_p)
        self.dropout_1 = torch.nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # MultiHeadedAttention (masked) - Sublayer 0
        attention = self.multi_headed_attention(x, x, x, mask)
        attention_dropout = self.dropout_0(attention)

        # Add + LayerNorm
        residual_0 = attention_dropout + x
        layer_norm_0 = self.layer_norm_0(residual_0)

        # FeedForward - Sublayer 1
        ff_0 = self.activation(self.feed_forward_0(layer_norm_0))
        ff_1 = self.feed_forward_1(ff_0)
        ff_1_dropout = self.dropout_1(ff_1)

        # Add + LayerNorm
        residual_1 = ff_1_dropout + layer_norm_0
        layer_norm_1 = self.layer_norm_1(residual_1)

        return layer_norm_1
