"""
Custom Qwen-2.5 model for masked discrete diffusion for ARC-AGI
Adapted from https://github.com/khalil-research/ViTARC/blob/main/vitarc/models/model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM

from src.models.embeddings import RelativePositionBias, HierarchicalPositionEncoding, ObjectPositionEncoding


class EnhancedAttention(nn.Module):
    """
    Enhanced attention module that incorporates relative position bias for 2D data.
    Based on the self-attention mechanism in transformers but with added positional awareness.
    """
    def __init__(self, config, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # Check that dimensions are compatible with number of heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Relative position bias
        self.relative_position_bias = RelativePositionBias(
            num_heads=num_heads,
            max_rows=getattr(config, "rows", 30),
            max_cols=getattr(config, "cols", 34),
            rpe_type=getattr(config, "rpe_type", "Two-slope-Alibi"),
            rpe_abs=getattr(config, "rpe_abs", True)
        )

        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_layer = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """Reshape for multi-head attention"""
        batch_size, seq_len = x.size()[:2]

        # Reshape: [batch, seq_len, d_model] -> [batch, seq_len, num_heads, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        relative_position=None,
        output_attentions=False
    ):
        """
        Args:
            hidden_states: Input of shape [batch_size, seq_len, d_model]
            attention_mask: Mask of shape [batch_size, 1, 1, seq_len]
            relative_position: Optional relative position information
            output_attentions: Whether to return attention weights

        Returns:
            context_layer: Attended output of shape [batch_size, seq_len, d_model]
            attention_probs: Optional attention probabilities
        """
        batch_size, seq_length = hidden_states.shape[:2]

        # Project inputs to queries, keys, and values
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Add relative position bias if provided
        if relative_position is not None:
            # Use the provided relative position information
            attention_scores = attention_scores + relative_position
        else:
            # Compute relative position bias based on sequence length
            position_bias = self.relative_position_bias.compute_bias(
                query_length=seq_length,
                key_length=seq_length,
                device=attention_scores.device
            )
            attention_scores = attention_scores + position_bias

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention probabilities
        attention_probs = self.dropout_layer(attention_probs)

        # Compute weighted sum of values based on attention probabilities
        context_layer = torch.matmul(attention_probs, value)

        # Reshape back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.d_model)

        # Final projection
        output = self.out_proj(context_layer)

        if output_attentions:
            return output, attention_probs
        else:
            return output


class CrossGridAttention(nn.Module):
    """
    Enhanced cross-grid attention mechanism to help the model relate patterns
    between input and output grids. Now incorporates relative position bias.
    """
    def __init__(self, config, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Use the enhanced attention for computing cross-grid relationships
        self.attention = EnhancedAttention(
            config=config,
            d_model=d_model,
            num_heads=num_heads
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, grid_info, attention_mask=None, relative_position=None):
        """
        Allow tokens in output grids to attend to corresponding input grids.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, d_model]
            grid_info: Dictionary containing grid metadata
            attention_mask: Optional attention mask
            relative_position: Optional relative position information

        Returns:
            Updated hidden states with cross-grid attention applied
        """
        batch_size, seq_len, d_model = hidden_states.size()

        # Create masks for input and output grid tokens
        is_input = (grid_info['grid_types'] == 0).unsqueeze(-1).expand(-1, -1, d_model)
        is_output = (grid_info['grid_types'] == 1).unsqueeze(-1).expand(-1, -1, d_model)

        # Also consider grid IDs to match corresponding input/output pairs
        grid_ids = grid_info['grid_ids'].unsqueeze(-1).expand(-1, -1, d_model)

        # Process each batch item separately to handle variable grid configurations
        results = []
        for b in range(batch_size):
            # For each output grid, attend to the corresponding input grid
            unique_grid_ids = torch.unique(grid_info['grid_ids'][b])

            # New hidden states for this batch
            new_h = hidden_states[b:b+1].clone()

            for grid_id in unique_grid_ids:
                # Skip special grid_id values (e.g., for special tokens)
                if grid_id == 0:  # Assuming 0 is reserved
                    continue

                # Find tokens belonging to this grid pair
                is_current_input = (grid_info['grid_ids'][b] == grid_id) & (grid_info['grid_types'][b] == 0)
                is_current_output = (grid_info['grid_ids'][b] == grid_id) & (grid_info['grid_types'][b] == 1)

                if not is_current_input.any() or not is_current_output.any():
                    continue

                # Extract hidden states for input and output grids
                output_hidden = hidden_states[b:b+1, is_current_output]
                input_hidden = hidden_states[b:b+1, is_current_input]

                # Create a cross-attention mask if needed
                cross_attn_mask = None
                if attention_mask is not None:
                    # Extract relevant parts of the attention mask
                    cross_attn_mask = attention_mask[b:b+1, :, is_current_output][:, :, :, is_current_input]

                # Apply attention to allow output to attend to input
                # Reshape for the attention mechanism
                output_hidden_flat = output_hidden.view(1, -1, d_model)
                input_hidden_flat = input_hidden.view(1, -1, d_model)

                # Prepare attention inputs
                attention_input = torch.cat([output_hidden_flat, input_hidden_flat], dim=1)

                # Call the attention mechanism
                attended_output = self.attention(
                    hidden_states=attention_input,
                    attention_mask=cross_attn_mask,
                    relative_position=relative_position
                )

                # Extract updated output tokens
                updated_output = attended_output[:, :output_hidden.size(1)]

                # Insert updated values back into the sequence
                output_indices = torch.where(is_current_output)[0]
                new_h[:, output_indices] = updated_output

            # Add this batch's result
            results.append(new_h)

        # Combine results
        result = torch.cat(results, dim=0)

        # Residual connection and normalization
        return self.norm(hidden_states + result)

#################################################
# MODEL ARCHITECTURE
#################################################


class ARCQwenModel(nn.Module):
    """
    Enhanced adaptation of Qwen model for ARC-AGI tasks with bidirectional attention,
    hierarchical position encoding, 2D position encoding, and object position encoding.
    """
    def __init__(self, pretrained_model_name='Qwen/Qwen1.5-1.8B', d_model=2048,
                 max_grids=10, max_grid_types=2, max_rows=30, max_cols=30,
                 train_from_scratch=True, add_cross_attn=True,
                 mixer_strategy="weighted_sum", use_2d_sinusoidal=True,
                 use_object_encoding=True, max_objects=100,
                 rpe_type="Two-slope-Alibi", rpe_abs=True):
        super().__init__()

        # Load model configuration
        self.config = AutoConfig.from_pretrained(pretrained_model_name)

        # Store position encoding configurations for reuse
        self.config.rows = max_rows
        self.config.cols = max_cols
        self.config.rpe_type = rpe_type
        self.config.rpe_abs = rpe_abs
        self.config.ape_mixer = mixer_strategy
        self.config.use_OPE = use_object_encoding

        # Adjust config for bidirectional attention
        self.config.use_cache = False
        self.config.is_decoder = False  # Disable causal masking

        # Update embedding dimension if needed
        self.d_model = d_model if d_model else self.config.hidden_size

        if train_from_scratch:
            # Create model with random initialization
            self.qwen = AutoModelForCausalLM.from_config(self.config)
        else:
            # Load pretrained model
            self.qwen = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

        # Create custom token embedding (using vocabulary for ARC)
        self.wte = nn.Embedding(self.config.vocab_size, self.d_model)

        # Replace the original word embeddings with our custom one
        self.qwen.model.embed_tokens = self.wte

        # Add enhanced hierarchical position encoding with 2D sinusoidal option
        self.hierarchical_pos_encoder = HierarchicalPositionEncoding(
            d_model=self.d_model,
            max_grids=max_grids,
            max_grid_types=max_grid_types,
            max_rows=max_rows,
            max_cols=max_cols,
            mixer_strategy=mixer_strategy,
            use_2d_sinusoidal=use_2d_sinusoidal
        )

        # Add object position encoding
        self.object_pos_encoder = ObjectPositionEncoding(
            d_model=self.d_model,
            max_objects=max_objects,
            enable_ope=use_object_encoding
        )

        # Cross-grid attention
        self.add_cross_attn = add_cross_attn
        if add_cross_attn:
            # Add cross-grid attention after each self-attention layer
            self.cross_attentions = nn.ModuleList([
                CrossGridAttention(
                    config=self.config,
                    d_model=self.d_model,
                    num_heads=self.config.num_attention_heads
                )
                for _ in range(0, self.config.num_hidden_layers, self.config.cross_attn_interval)
            ])

        # Relative position bias to be used globally
        self.relative_position_bias = RelativePositionBias(
            num_heads=self.config.num_attention_heads,
            max_rows=max_rows,
            max_cols=max_cols,
            rpe_type=rpe_type,
            rpe_abs=rpe_abs
        )

        # Initialize or reset weights for training from scratch
        if train_from_scratch:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights for training from scratch"""
        if isinstance(module, nn.Linear):
            # Slightly higher std than standard initialization for better gradient flow
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def prepare_inputs_embeds(self, input_ids, grid_info, object_idx=None):
        """
        Create input embeddings with hierarchical position encoding and object encoding.

        Args:
            input_ids: Token ids [batch_size, seq_len]
            grid_info: Dictionary with grid metadata
            object_idx: Optional object indices for object position encoding

        Returns:
            Input embeddings with position information
        """
        # Get token embeddings
        inputs_embeds = self.wte(input_ids)

        # Get hierarchical position encodings
        position_encodings = self.hierarchical_pos_encoder(
            grid_info['grid_ids'],
            grid_info['grid_types'],
            grid_info['row_positions'],
            grid_info['col_positions']
        )

        # Add object position encoding if enabled and provided
        if object_idx is not None and self.config.use_OPE:
            position_encodings = self.object_pos_encoder(object_idx, position_encodings)

        # Add position encodings to token embeddings
        return inputs_embeds + position_encodings

    def forward(self, input_ids=None, attention_mask=None, grid_info=None,
                object_idx=None, labels=None, output_attentions=False, **kwargs):
        """
        Forward pass with bidirectional attention, hierarchical position encoding,
        relative position bias, and object position encoding.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            grid_info: Dictionary with grid position information
            object_idx: Optional object indices for object position encoding
            labels: Optional labels for loss calculation
            output_attentions: Whether to output attention matrices

        Returns:
            Model outputs
        """
        # Create embeddings with hierarchical positions and object information
        inputs_embeds = self.prepare_inputs_embeds(input_ids, grid_info, object_idx)

        # Disable causal attention
        kwargs['use_cache'] = False

        # Get sequence length for relative position bias
        batch_size, seq_len = input_ids.shape

        # Pre-compute relative position bias based on sequence length
        relative_position_bias = self.relative_position_bias.compute_bias(
            query_length=seq_len,
            key_length=seq_len,
            device=inputs_embeds.device
        )

        # Create a function hook to modify attention masks and add relative position bias
        def modify_attention_hook(module, inputs, outputs):
            # Get the attention scores
            attention_scores = outputs[0]

            # Add relative position bias
            attention_scores = attention_scores + relative_position_bias

            # If there's an existing attention mask, respect it
            if attention_mask is not None:
                # Expand attention_mask to match the expected shape
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                expanded_mask = (1.0 - expanded_mask) * -10000.0
                attention_scores = attention_scores + expanded_mask

            # Replace outputs with modified attention scores
            return (attention_scores,) + outputs[1:]

        # Register hooks to modify attention in each layer
        hooks = []
        for layer in self.qwen.transformer.h:
            # Find the attention layer
            attn_layer = layer.attn if hasattr(layer, 'attn') else layer.attention

            # Register forward hook to modify attention scores
            hook = attn_layer.register_forward_hook(modify_attention_hook)
            hooks.append(hook)

        # Forward through base model
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=None,  # We'll handle this in the hook
            labels=labels,
            output_attentions=output_attentions,
            **kwargs
        )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Apply cross-grid attention if enabled
        if self.add_cross_attn and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = list(outputs.hidden_states)

            for i, cross_attn in enumerate(self.cross_attentions):
                # Apply cross-grid attention after each layer
                hidden_states[i+1] = cross_attn(
                    hidden_states[i+1],
                    grid_info,
                    attention_mask=attention_mask,
                    relative_position=relative_position_bias
                )

            # Update last hidden state
            outputs.last_hidden_state = hidden_states[-1]

        return outputs

#################################################
# TRAINING UTILITIES
#################################################


def diffusion_loss(model, input_ids, grid_info, mask_ratio=0.15, mask_token_id=17):
    """
    Compute the masked diffusion loss for ARC tasks.

    Args:
        model: The ARCQwenModel
        input_ids: Input token IDs
        grid_info: Grid metadata
        mask_ratio: Percentage of output grid tokens to mask
        mask_token_id: ID of the mask token

    Returns:
        Loss value
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Create random masking pattern for output grid cells
    output_grid_mask = (grid_info['grid_types'] == 1)  # Output grid tokens

    # Randomly select tokens to mask within output grid
    mask_probs = torch.rand(batch_size, seq_len, device=device)
    mask_indices = (mask_probs < mask_ratio) & output_grid_mask

    # Save original tokens for loss calculation
    original_tokens = input_ids.clone()

    # Replace masked tokens with mask token ID
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_indices] = mask_token_id

    # Forward pass
    outputs = model(input_ids=masked_input_ids, grid_info=grid_info)
    logits = outputs.logits

    # Calculate loss only on masked tokens
    loss_fct = nn.CrossEntropyLoss()
    masked_lm_loss = loss_fct(
        logits[mask_indices].view(-1, model.config.vocab_size),
        original_tokens[mask_indices].view(-1)
    )

    return masked_lm_loss

#################################################
# DATA PROCESSING
#################################################


def collate_arc_batch(batch):
    """Collate function for ARC examples with padding"""
    # Find max length in batch
    max_length = max(x['input_ids'].size(0) for x in batch)

    # Initialize padded tensors
    batch_size = len(batch)
    input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    grid_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    grid_types = torch.zeros(batch_size, max_length, dtype=torch.long)
    row_positions = torch.zeros(batch_size, max_length, dtype=torch.long)
    col_positions = torch.zeros(batch_size, max_length, dtype=torch.long)

    # Check if we have object indices
    has_object_idx = 'object_idx' in batch[0]
    if has_object_idx:
        object_idx = torch.zeros(batch_size, max_length, dtype=torch.long)

    # Pad each example
    for i, example in enumerate(batch):
        seq_len = example['input_ids'].size(0)
        input_ids[i, :seq_len] = example['input_ids']
        attention_mask[i, :seq_len] = example['attention_mask']
        grid_ids[i, :seq_len] = example['grid_ids']
        grid_types[i, :seq_len] = example['grid_types']
        row_positions[i, :seq_len] = example['row_positions']
        col_positions[i, :seq_len] = example['col_positions']
        if has_object_idx:
            object_idx[i, :seq_len] = example['object_idx']

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'grid_info': {
            'grid_ids': grid_ids,
            'grid_types': grid_types,
            'row_positions': row_positions,
            'col_positions': col_positions
        }
    }

    if has_object_idx:
        result['object_idx'] = object_idx

    return result

#################################################
# TRAINING LOOP
#################################################


def train_arc_model(model, train_dataloader, optimizer, scheduler=None,
                    num_epochs=10, device='cuda', mask_ratio=0.15,
                    mask_token_id=17, eval_dataloader=None, use_object_encoding=False):
    """
    Train the ARC model using masked diffusion.

    Args:
        model: ARCQwenModel
        train_dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        device: Device to train on
        mask_ratio: Ratio of tokens to mask in output grids
        mask_token_id: ID of the mask token
        eval_dataloader: Evaluation data loader (optional)
        use_object_encoding: Whether to use object position encoding

    Returns:
        Trained model and training statistics
    """
    model.to(device)
    model.train()

    stats = {
        'train_losses': [],
        'eval_losses': [] if eval_dataloader else None
    }

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            grid_info = {k: v.to(device) for k, v in batch['grid_info'].items()}

            # Generate object indices if using object encoding
            object_idx = None
            if use_object_encoding and 'object_idx' in batch:
                object_idx = batch['object_idx'].to(device)
            elif use_object_encoding:
                # Generate random object indices for training
                # In a real implementation, these would come from actual object detection
                object_idx = torch.randint(0, 50, (input_ids.size(0), input_ids.size(1)), device=device)

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss with diffusion
            batch_size, seq_len = input_ids.shape

            # Create random masking pattern for output grid cells
            output_grid_mask = (grid_info['grid_types'] == 1)  # Output grid tokens

            # Randomly select tokens to mask within output grid
            mask_probs = torch.rand(batch_size, seq_len, device=device)
            mask_indices = (mask_probs < mask_ratio) & output_grid_mask

            # Save original tokens for loss calculation
            original_tokens = input_ids.clone()

            # Replace masked tokens with mask token ID
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_indices] = mask_token_id

            # Forward pass
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                grid_info=grid_info,
                object_idx=object_idx
            )

            # Calculate loss only on masked tokens
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                logits[mask_indices].view(-1, model.config.vocab_size),
                original_tokens[mask_indices].view(-1)
            )

            # Backward and optimize
            masked_lm_loss.backward()
            optimizer.step()

            # Update stats
            epoch_loss += masked_lm_loss.item()

            # Log progress
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step}, Loss: {masked_lm_loss.item():.4f}")

        # Average epoch loss
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        stats['train_losses'].append(avg_epoch_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.4f}")

        # Evaluate if eval_dataloader is provided
        if eval_dataloader:
            eval_loss = evaluate_arc_model(
                model, eval_dataloader, device,
                mask_ratio=mask_ratio,
                mask_token_id=mask_token_id,
                use_object_encoding=use_object_encoding
            )
            stats['eval_losses'].append(eval_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Eval Loss: {eval_loss:.4f}")

        # Update learning rate
        if scheduler:
            scheduler.step()

    return model, stats


def evaluate_arc_model(model, dataloader, device='cuda', mask_ratio=0.15, mask_token_id=17, use_object_encoding=False):
    """
    Evaluate the ARC model.

    Args:
        model: ARCQwenModel
        dataloader: Evaluation data loader
        device: Device to evaluate on
        mask_ratio: Ratio of tokens to mask in output grids
        mask_token_id: ID of the mask token
        use_object_encoding: Whether to use object position encoding

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            grid_info = {k: v.to(device) for k, v in batch['grid_info'].items()}

            # Generate object indices if using object encoding
            object_idx = None
            if use_object_encoding and 'object_idx' in batch:
                object_idx = batch['object_idx'].to(device)
            elif use_object_encoding:
                # Generate random object indices for evaluation
                object_idx = torch.randint(0, 50, (input_ids.size(0), input_ids.size(1)), device=device)

            # Create random masking pattern for output grid cells
            batch_size, seq_len = input_ids.shape
            output_grid_mask = (grid_info['grid_types'] == 1)  # Output grid tokens

            # Randomly select tokens to mask within output grid
            mask_probs = torch.rand(batch_size, seq_len, device=device)
            mask_indices = (mask_probs < mask_ratio) & output_grid_mask

            # Save original tokens for loss calculation
            original_tokens = input_ids.clone()

            # Replace masked tokens with mask token ID
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_indices] = mask_token_id

            # Forward pass
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                grid_info=grid_info,
                object_idx=object_idx
            )

            # Calculate loss only on masked tokens
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                logits[mask_indices].view(-1, model.config.vocab_size),
                original_tokens[mask_indices].view(-1)
            )

            total_loss += masked_lm_loss.item()

    return total_loss / len(dataloader)

#################################################
# INFERENCE
#################################################


def diffusion_inference(model, input_grid, tokenizer, device='cuda', num_steps=64, object_idx=None):
    """
    Generate output grid using enhanced diffusion-based inference.

    Args:
        model: ARCQwenModel
        input_grid: Input grid (2D list)
        tokenizer: Tokenizer
        device: Device to run inference on
        num_steps: Number of diffusion steps
        object_idx: Optional object indices for object position encoding

    Returns:
        Generated output grid
    """
    model.eval()

    # Create input example with dummy output grid
    dummy_output = [[0 for _ in range(len(input_grid[0]))] for _ in range(len(input_grid))]
    example = ARCExample(input_grid=input_grid, output_grid=dummy_output, grid_id=1)

    # Create dataset with single example
    dataset = ARCDataset([example], tokenizer)
    example_batch = dataset[0]

    # Move to device
    input_ids = example_batch['input_ids'].unsqueeze(0).to(device)
    attention_mask = example_batch['attention_mask'].unsqueeze(0).to(device)
    grid_info = {
        k: v.unsqueeze(0).to(device)
        for k, v in {
            'grid_ids': example_batch['grid_ids'],
            'grid_types': example_batch['grid_types'],
            'row_positions': example_batch['row_positions'],
            'col_positions': example_batch['col_positions']
        }.items()
    }

    # Prepare object indices if provided
    if object_idx is not None:
        object_idx = object_idx.unsqueeze(0).to(device)

    # Find output grid tokens
    is_output = (grid_info['grid_types'] == 1) & (grid_info['row_positions'] >= 0) & (grid_info['col_positions'] >= 0)

    # Get mask token ID
    mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    # Initially mask all output tokens
    masked_input_ids = input_ids.clone()
    masked_input_ids[is_output] = mask_token_id

    # Progressive denoising
    for step in range(num_steps):
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                grid_info=grid_info,
                object_idx=object_idx
            )

        # Get predictions for output tokens
        logits = outputs.logits

        # Find current masked positions
        current_masks = (masked_input_ids == mask_token_id)

        # Progressively reduce temperature
        temperature = max(0.1, 1.0 - step / num_steps)

        if step < num_steps - 1:
            # Sample with temperature
            probs = F.softmax(logits[current_masks] / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)
        else:
            # Final step - use argmax
            next_tokens = torch.argmax(logits[current_masks], dim=-1)

        # Update masked inputs
        masked_input_ids[current_masks] = next_tokens

        # For progressive denoising, re-mask a decreasing portion of tokens
        if step < num_steps - 1:
            unmask_ratio = (step + 1) / num_steps

            # Calculate how many tokens to keep unmasked
            num_output_tokens = is_output.sum().item()
            num_to_unmask = int(unmask_ratio * num_output_tokens)

            # Randomly select tokens to mask
            output_mask_candidates = is_output.clone()
            flat_indices = torch.arange(output_mask_candidates.numel())[output_mask_candidates.view(-1)]
            perm = torch.randperm(len(flat_indices))
            to_mask = flat_indices[perm[num_to_unmask:]]

            # Create a mask tensor of the same shape as masked_input_ids
            new_mask = torch.zeros_like(masked_input_ids, dtype=torch.bool)
            new_mask.view(-1)[to_mask] = True

            # Apply the mask
            masked_input_ids[new_mask] = mask_token_id

    # Extract output grid
    output_token_indices = torch.nonzero(is_output.squeeze(0))
    output_rows = grid_info['row_positions'][0][output_token_indices[:, 0]].cpu().numpy()
    output_cols = grid_info['col_positions'][0][output_token_indices[:, 0]].cpu().numpy()
    output_values = masked_input_ids[0][output_token_indices[:, 0]].cpu().numpy()

    # Create output grid
    max_row = int(output_rows.max()) + 1
    max_col = int(output_cols.max()) + 1
    output_grid = [[0 for _ in range(max_col)] for _ in range(max_row)]

    for row, col, val in zip(output_rows, output_cols, output_values):
        output_grid[int(row)][int(col)] = int(val)

    return output_grid


#################################################
# MAIN EXAMPLE
#################################################


def main():
    """Example usage of the enhanced ARC-AGI implementation"""
    # Create tokenizer
    tokenizer_config = {}
    tokenizer = create_arc_tokenizer(tokenizer_config)

    # Configuration for enhanced model
    model_config = {
        'pretrained_model_name': 'Qwen/Qwen1.5-1.8B',
        'd_model': 2048,
        'max_grids': 10,
        'max_rows': 30,
        'max_cols': 30,
        'train_from_scratch': True,
        'add_cross_attn': True,
        'mixer_strategy': 'weighted_sum',
        'use_2d_sinusoidal': True,
        'use_object_encoding': True,
        'max_objects': 100,
        'rpe_type': 'Two-slope-Alibi',
        'rpe_abs': True
    }

    # Create enhanced model
    model = ARCQwenModel(**model_config)

    # Create some example data with object patterns
    examples = [
        ARCExample(
            input_grid=[
                [0, 1, 0],
                [1, 2, 1],
                [0, 1, 0]
            ],
            output_grid=[
                [1, 0, 1],
                [0, 2, 0],
                [1, 0, 1]
            ],
            grid_id=1
        ),
        ARCExample(
            input_grid=[
                [1, 1, 3],
                [1, 0, 3],
                [3, 3, 3]
            ],
            output_grid=[
                [0, 0, 3],
                [0, 1, 3],
                [3, 3, 3]
            ],
            grid_id=2
        )
    ]

    # Create dataset with object extraction
    dataset = ARCDataset(examples, tokenizer, extract_objects=True)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_arc_batch
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Enhanced Model Architecture:")
    print(model)

    print("\nStarting training with enhanced features...")
    # In practice, you would train for more epochs and use proper validation
    model, stats = train_arc_model(
        model,
        dataloader,
        optimizer,
        num_epochs=1,  # Just for demonstration
        device=device,
        use_object_encoding=True
    )

    print("Training complete!")

    # Generate output for a new input
    new_input = [[1, 0, 1], [0, 3, 0], [1, 0, 1]]

    # Extract objects for inference
    object_extractor = ARCDataset([], tokenizer, extract_objects=True)
    object_map = object_extractor.extract_grid_objects(new_input)

    # Convert object map to tensor
    object_tensor = torch.tensor([0] + [obj for row in object_map for obj in row] + [0], dtype=torch.long)

    # Generate output using enhanced model
    output_grid = diffusion_inference(
        model,
        new_input,
        tokenizer,
        device,
        object_idx=object_tensor
    )

    print("Generated output grid:")
    for row in output_grid:
        print(row)

    # Visualize attention patterns
    print("\nVisualizing attention patterns...")
    attn_weight, batch_data = visualize_attention(model, new_input, tokenizer, device=device)
    print(f"Attention matrix shape: {attn_weight.shape}")

    # Visualize relative position bias
    print("\nVisualizing relative position bias...")
    bias_grids = visualize_relative_position_bias(model, max_rows=5, max_cols=5)
    for head_name, bias_grid in list(bias_grids.items())[:2]:  # Show first two heads only
        print(f"{head_name} bias pattern shape: {bias_grid.shape}")

    # Visualize object embeddings
    print("\nVisualizing object embeddings...")
    obj_embeddings = visualize_object_embeddings(model)
    if obj_embeddings:
        for obj_id, embedding in list(obj_embeddings.items())[:2]:  # Show first two objects only
            print(f"{obj_id} embedding shape: {embedding.shape}")

    print("\nEnhanced model implementation complete!")


if __name__ == "__main__":
    main()
