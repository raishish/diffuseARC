import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectPositionEncoding(nn.Module):
    """
    Object Position Encoding (OPE) to add "objectness" information to position embeddings

    Object identity embeddings can be obtained from an object recognition model.
    """
    def __init__(self, d_model, max_objects=100, enable_ope=True):
        super().__init__()
        self.d_model = d_model
        self.enable_ope = enable_ope

        if enable_ope:
            # Object identity embedding takes half the embedding dimension
            self.obj_embedding = nn.Embedding(max_objects, d_model // 2)

            # Initialize weights
            nn.init.xavier_normal_(self.obj_embedding.weight)

            # Layer norm for combining with position embeddings
            self.norm = nn.LayerNorm(d_model)

    def forward(self, object_idx, position_embeds):
        """
        Args:
            object_idx: Tensor of shape [batch_size, seq_len] with object indices
            position_embeds: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Combined position and object embeddings [batch_size, seq_len, d_model]
        """
        if not self.enable_ope or object_idx is None:
            return position_embeds

        # Get object embeddings (half dimensional)
        obj_embeds = self.obj_embedding(object_idx)  # [batch_size, seq_len, d_model//2]

        # Split position embeddings into two halves
        pos_embeds_half = position_embeds[..., :self.d_model//2]  # [batch_size, seq_len, d_model//2]

        # Combine object embeddings with the first half of position embeddings
        combined_embeds = torch.cat([obj_embeds, pos_embeds_half], dim=-1)

        # Apply layer normalization
        output_embeds = self.norm(combined_embeds)

        return output_embeds


class RelativePositionBias(nn.Module):
    """
    Implements Alibi-style relative position bias for 2D grids.
    Adapted from: https://github.com/ofirpress/attention_with_linear_biases

    Supported types:
      - "NoRPE": No relative position bias
      - "Two-slope-Alibi": Two slopes for left and right directions
      - "Four-diag-slope-Alibi": Four slopes for diagonal directions
    """
    def __init__(self, num_heads, max_rows=30, max_cols=30, rpe_type="Two-slope-Alibi", rpe_abs=True):
        super().__init__()
        self.num_heads = num_heads
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.rpe_type = rpe_type
        self.rpe_abs = rpe_abs

        # Pre-calculate the distance matrix for the 2D grid
        distance_matrix = self.calculate_2d_relative_positions(max_rows, max_cols)
        self.register_buffer('distance_matrix', distance_matrix)

        # Generate slopes for attention heads
        if rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
            slopes_l = torch.tensor(self.get_slopes(num_heads, start_exponent=1)) * -1
            slopes_r = torch.tensor(self.get_slopes(num_heads, start_exponent=0.5)) * -1
            self.register_buffer('slopes_l', slopes_l)
            self.register_buffer('slopes_r', slopes_r)

    def get_slopes(self, n, start_exponent=1):
        """
        Get geometric slopes for Alibi bias.
        Based on the original ALiBi paper.
        """
        def get_geometric_slopes(n, start_exponent):
            start = 2 ** (-start_exponent)  # Starting value 2^(-start_exponent)
            ratio = 2 ** -1  # Halving each step
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n).is_integer():
            return get_geometric_slopes(n, start_exponent)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (get_geometric_slopes(closest_power_of_2, start_exponent) +
                    self.get_slopes(2 * closest_power_of_2, start_exponent)[0::2][:n - closest_power_of_2])

    def calculate_2d_relative_positions(self, grid_height, grid_width):
        """
        Calculate relative positions for a 2D grid based on Manhattan distance.
        For some directions, apply a factor to adjust the distance.
        """
        if self.rpe_type == "Four-diag-slope-Alibi":
            # Define direction-specific factors
            top_right_factor = 2 ** 0.25
            down_right_factor = 2 ** 0.25
        else:
            top_right_factor = 1.0
            down_right_factor = 1.0

        # Create grid coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(grid_height),
            torch.arange(grid_width),
            indexing='ij'
        )

        # Flatten the 2D grid coordinates
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()

        # Calculate total number of positions in the grid
        grid_size = grid_height * grid_width

        # Initialize relative position matrix
        relative_position = torch.zeros((grid_size, grid_size), dtype=torch.float)

        # Calculate Manhattan distance between each pair of points
        for i in range(grid_size):
            for j in range(grid_size):
                x_diff = x_flat[i] - x_flat[j]
                y_diff = y_flat[i] - y_flat[j]
                manhattan_distance = float(abs(x_diff) + abs(y_diff))

                # Adjust distance based on direction
                if x_diff < 0 and y_diff < 0:  # Top-right
                    manhattan_distance *= top_right_factor
                elif x_diff > 0 and y_diff < 0:  # Down-right
                    manhattan_distance *= down_right_factor

                relative_position[i, j] = manhattan_distance

        return relative_position

    def compute_bias(self, query_length, key_length, device=None):
        """
        Compute the relative position bias for attention scores.

        Args:
            query_length: Length of query sequence
            key_length: Length of key sequence
            device: Device to put the bias on

        Returns:
            Attention bias of shape [1, num_heads, query_length, key_length]
        """
        if device is None:
            device = self.distance_matrix.device

        if self.rpe_type == "NoRPE":
            # No relative position bias
            return torch.zeros((1, self.num_heads, query_length, key_length), device=device)

        elif self.rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
            # Get the relevant part of the distance matrix
            relative_position = self.distance_matrix[:query_length, :key_length].to(device)

            if self.rpe_abs:
                # Use absolute distances
                relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.num_heads, -1, -1)
            else:
                # Use signed distances
                relative_position = relative_position.unsqueeze(0).expand(self.num_heads, -1, -1)

            # Apply head-specific slopes based on the relative positions
            self.slopes_l = self.slopes_l.to(device)
            self.slopes_r = self.slopes_r.to(device)

            # Use different slopes for different parts of the relative positions
            alibi_left = self.slopes_l.unsqueeze(1).unsqueeze(1) * relative_position
            alibi_right = self.slopes_r.unsqueeze(1).unsqueeze(1) * relative_position

            # Combine the biases based on the position relationships
            values = torch.triu(alibi_right) + torch.tril(alibi_left)

            # Reshape to the required output form
            values = values.view(1, self.num_heads, query_length, key_length)

            return values

        else:
            # Default case: no bias
            return torch.zeros((1, self.num_heads, query_length, key_length), device=device)


class FixedAbsolutePositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings from the original Transformer paper (Vaswani et al., 2017).
    """
    def __init__(self, dim, max_positions=16384):
        super().__init__()
        # Calculate frequencies for each dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        # Create position indices
        t = torch.arange(max_positions).type_as(inv_freq)

        # Calculate sinusoidal inputs
        sinusoid_inp = torch.einsum("i,j -> ij", t, inv_freq)

        # Combine sin and cos values
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

        # If dimension is odd, we need to handle the last dimension separately
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(max_positions, 1)], dim=-1)

        # Create embedding layer with pre-calculated weights
        self.embedding = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids):
        """
        Args:
            position_ids: Tensor of position indices [batch_size, seq_len]

        Returns:
            Positional embeddings [batch_size, seq_len, dim]
        """
        return self.embedding(position_ids.long())


class EmbeddingMixer(nn.Module):
    """
    A module for mixing input embeddings and positional embeddings according to different strategies.

    Supported strategies:
      - 'hardcoded_normalization'
      - 'learnable_scaling'
      - 'weighted_sum'
      - 'weighted_sum_no_norm'
      - 'learnable_scaling_vec'
      - 'weighted_sum_vec'
      - 'weighted_sum_no_norm_vec'
      - 'positional_attention'
      - 'layer_norm'
      - 'default'
    """
    def __init__(self, embed_dim: int, mixer_strategy: str = "default"):
        """
        Args:
            embed_dim: Dimension of the embeddings
            mixer_strategy: Strategy for mixing embeddings
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mixer_strategy = mixer_strategy

        # Vector-based parameters for element-wise operations
        if self.mixer_strategy in ['learnable_scaling_vec', 'weighted_sum_vec', 'weighted_sum_no_norm_vec']:
            self.position_scale = nn.Parameter(torch.ones(1, embed_dim))
            self.input_weight = nn.Parameter(torch.ones(1, embed_dim))
            self.position_weight = nn.Parameter(torch.ones(1, embed_dim))

        # Scalar parameters for global operations
        if self.mixer_strategy in ['learnable_scaling', 'weighted_sum', 'weighted_sum_no_norm']:
            self.position_scale = nn.Parameter(torch.ones(1))
            self.input_weight = nn.Parameter(torch.ones(1))
            self.position_weight = nn.Parameter(torch.ones(1))

        # Attention-based mixing
        if self.mixer_strategy == 'positional_attention':
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)

        # Layer norm for combined embeddings
        if self.mixer_strategy == 'layer_norm':
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs_embeds: torch.Tensor, position_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs_embeds: Token embeddings [batch_size, seq_len, embed_dim]
            position_embeds: Position embeddings [batch_size, seq_len, embed_dim]

        Returns:
            Combined embeddings [batch_size, seq_len, embed_dim]
        """
        strategy = self.mixer_strategy

        if strategy == 'hardcoded_normalization':
            # Normalize both input and position embeddings
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = inputs_embeds_norm + position_embeds_norm

        elif strategy in ['learnable_scaling', 'learnable_scaling_vec']:
            # Scale positional embeddings with a learnable parameter
            scaled_position_embeds = self.position_scale * position_embeds
            output_embeds = inputs_embeds + scaled_position_embeds

        elif strategy in ['weighted_sum', 'weighted_sum_vec']:
            # Learnable weighted sum of normalized embeddings
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = (self.input_weight * inputs_embeds_norm) + (self.position_weight * position_embeds_norm)

        elif strategy in ['weighted_sum_no_norm', 'weighted_sum_no_norm_vec']:
            # Learnable weighted sum without normalization
            output_embeds = (self.input_weight * inputs_embeds) + (self.position_weight * position_embeds)

        elif strategy == 'positional_attention':
            # Use attention to combine input and position
            position_embeds_expanded = position_embeds.expand(inputs_embeds.shape[0], -1, -1)

            # Reshape for MultiheadAttention
            inputs_embeds_reshaped = inputs_embeds.transpose(0, 1)
            position_embeds_reshaped = position_embeds_expanded.transpose(0, 1)

            attn_output, _ = self.attention(
                inputs_embeds_reshaped,
                position_embeds_reshaped,
                position_embeds_reshaped
            )
            output_embeds = inputs_embeds_reshaped + attn_output
            output_embeds = output_embeds.transpose(0, 1)  # back to [batch_size, seq_len, embed_dim]

        elif strategy == 'layer_norm':
            # Simple addition with layer normalization
            combined_embeds = inputs_embeds + position_embeds
            output_embeds = self.layer_norm(combined_embeds)

        elif strategy == 'default':
            # Simple addition without any special treatment
            output_embeds = inputs_embeds + position_embeds

        else:
            raise ValueError(f"Unsupported mixer_strategy: {strategy}")

        return output_embeds


class HierarchicalPositionEncoding(nn.Module):
    """
    Hierarchical position encoding for ARC tasks with multiple grids.

    Supported encodings:
        - grid_id
        - grid_type (input/output)
        - row_position
        - col_position

    Now with 2D sinusoidal position encodings and flexible embedding mixing.
    """
    def __init__(self, d_model, max_grids=10, max_grid_types=2, max_rows=30, max_cols=30,
                 mixer_strategy="weighted_sum", dropout=0.1, use_2d_sinusoidal=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.use_2d_sinusoidal = use_2d_sinusoidal

        # 2D spatial position embeddings
        if use_2d_sinusoidal:
            # Use fixed sinusoidal embeddings
            self.row_embedding = FixedAbsolutePositionalEmbedding(d_model // 2, max_positions=max_rows)
            self.col_embedding = FixedAbsolutePositionalEmbedding(d_model // 2, max_positions=max_cols)
        else:
            # Use learned embeddings
            self.row_embedding = nn.Embedding(max_rows, d_model)
            self.col_embedding = nn.Embedding(max_cols, d_model)
            nn.init.xavier_normal_(self.row_embedding.weight)
            nn.init.xavier_normal_(self.col_embedding.weight)

        # Grid embeddings
        self.grid_id_embedding = nn.Embedding(max_grids, d_model)
        self.grid_type_embedding = nn.Embedding(max_grid_types, d_model)  # 0=input, 1=output
        nn.init.xavier_normal_(self.grid_id_embedding.weight)
        nn.init.xavier_normal_(self.grid_type_embedding.weight)

        # Embedding mixer for combining different embeddings
        self.grid_mixer = EmbeddingMixer(d_model, mixer_strategy)
        self.pos_mixer = EmbeddingMixer(d_model, mixer_strategy)

        # Learnable weights for combining the hierarchical components
        self.combine_weights = nn.Parameter(torch.ones(4) / 4)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, grid_ids, grid_types, row_positions, col_positions):
        """
        Forward pass to compute hierarchical position encodings.

        Args:
            grid_ids: Tensor of shape [batch_size, seq_len] with grid indices
            grid_types: Tensor of shape [batch_size, seq_len] with grid types (0=input, 1=output)
            row_positions: Tensor of shape [batch_size, seq_len] with row positions
            col_positions: Tensor of shape [batch_size, seq_len] with column positions

        Returns:
            position_encodings: Tensor of shape [batch_size, seq_len, d_model]
        """
        # Get embeddings for each component
        grid_id_embeddings = self.grid_id_embedding(grid_ids)
        grid_type_embeddings = self.grid_type_embedding(grid_types)

        # Get spatial position embeddings
        row_embeddings = self.row_embedding(row_positions)
        col_embeddings = self.col_embedding(col_positions)

        # Combine row and column embeddings
        if self.use_2d_sinusoidal:
            # For sinusoidal embeddings, we concatenate them
            pos_embeddings = torch.cat([row_embeddings, col_embeddings], dim=-1)
        else:
            # For learned embeddings, we can use the mixer
            pos_embeddings = self.pos_mixer(row_embeddings, col_embeddings)

        # Combine grid and type embeddings
        grid_embeddings = self.grid_mixer(grid_id_embeddings, grid_type_embeddings)

        # Normalize combination weights with softmax
        norm_weights = torch.softmax(self.combine_weights, dim=0)

        # Weighted combination of grid and position embeddings
        position_encodings = norm_weights[0] * grid_embeddings + norm_weights[1] * pos_embeddings

        # Apply normalization and dropout
        position_encodings = self.norm(position_encodings)
        position_encodings = self.dropout(position_encodings)

        return position_encodings
