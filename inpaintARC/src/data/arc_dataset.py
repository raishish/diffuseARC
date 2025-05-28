import torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset
import numpy as np
from transformers import PreTrainedTokenizer
from typing import List, Any


@dataclass(frozen=True)
class ARCGrid:
    """Class for storing an ARC grid."""
    grid: List[List[int]]
    example_id: int  # example idx
    grid_type: int   # 0 for input, 1 for output

    def __post_init__(self):
        # Validate that grid is a 2D list
        if not self.grid or not isinstance(self.grid[0], list):
            raise ValueError("Grid must be a 2D list")
        # Validate that all rows have the same length
        if not all(len(row) == len(self.grid[0]) for row in self.grid):
            raise ValueError("All rows in the grid must have the same length")
        # Validate that grid contains only integers
        if not all(isinstance(cell, int) for row in self.grid for cell in row):
            raise ValueError("Grid must contain only integers")

    @property
    def shape(self):
        """Get the shape of the grid."""
        return len(self.grid), len(self.grid[0]) if self.grid else (0, 0)

    def pad_grid(
        self, tokenizer: PreTrainedTokenizer, max_rows: int, max_cols: int, mask: bool = False
    ) -> tuple:
        """Pad the grid to the specified size.

        Args:
            tokenizer: PreTrainedTokenizer for tokenization
            max_rows: maximum number of rows
            max_cols: maximum number of columns
            mask: whether to mask the grid (default: False)
        Returns:
            padded_grid: padded grid with the specified number of rows and columns
            padded_example_ids: padded grid filled with example_id
            padded_grid_types: padded grid filled with grid_type
        """
        padded_grid = []
        rows, cols = len(self.grid), len(self.grid[0])
        required_padding_rows = max_rows - rows
        required_padding_cols = max_cols - cols

        # Add padding columns
        for row in self.grid:
            if mask:
                padded_grid.append(
                    [tokenizer.mask_token] * rows +
                    [tokenizer.arc_grid_endx_token] +
                    [tokenizer.arc_pad_token] * required_padding_cols +
                    [tokenizer.arc_sep_row_token]
                )
            else:
                padded_grid.append(
                    row +
                    [tokenizer.arc_grid_endx_token] +
                    [tokenizer.arc_pad_token] * required_padding_cols +
                    [tokenizer.arc_sep_row_token]
                )

        # Add column boundary tokens
        padded_grid.append(
            [tokenizer.arc_grid_endy_token] * cols +
            [tokenizer.arc_grid_endxy_token] +
            [tokenizer.arc_grid_endy_token] * required_padding_cols +
            [tokenizer.arc_sep_row_token]
        )

        # Add padding rows
        for _ in range(required_padding_rows):
            padded_grid.append(
                [tokenizer.arc_pad_token] * cols +
                [tokenizer.arc_grid_endx_token] +
                [tokenizer.arc_pad_token] * required_padding_cols +
                [tokenizer.arc_sep_row_token]
            )

        padded_grid = np.array(padded_grid)
        padded_example_ids = np.full(padded_grid.shape, self.example_id)
        padded_grid_types = np.full(padded_grid.shape, self.grid_type)

        return (padded_grid, padded_example_ids, padded_grid_types)

    def unpad_grid(self, padded_grid: List[List[Any]], tokenizer: PreTrainedTokenizer) -> List[List[int]]:
        """Unpad the grid to its original size.

        Args:
            padded_grid: padded grid to unpad
        Returns:
            unpadded_grid: unpadded grid with the original size
        """
        unpadded_grid = []
        special_tokens = list(tokenizer.special_tokens_maps.keys())

        for row in padded_grid:
            if row[0] == tokenizer.arc_pad_token:
                continue
            unpadded_row = [int(cell) for cell in row if cell not in special_tokens]
            unpadded_grid.append(unpadded_row)

        # Validate that grid contains only integers
        if not all(isinstance(cell, int) for row in unpadded_grid for cell in row):
            raise ValueError("Unpadded grid must contain only integers")
        # Validate that all rows have the same length
        if not all(len(row) == len(unpadded_grid[0]) for row in unpadded_grid):
            raise ValueError("All rows in the unpadded grid must have the same length")
        # Validate that grid is a 2D list
        if not unpadded_grid or not isinstance(unpadded_grid[0], list):
            raise ValueError("Unpadded grid must be a 2D list")

        return unpadded_grid


@dataclass
class ARCExample:
    """Class for storing an ARC task example."""
    task_id: str
    input_grid: ARCGrid
    output_grid: ARCGrid
    example_id: int    # example idx
    example_type: int  # 0 for train, 1 for test

    @property
    def input_rows(self):
        return len(self.input_grid.grid)

    @property
    def input_cols(self):
        return len(self.input_grid.grid[0]) if self.input_grid.grid else 0

    @property
    def output_rows(self):
        return len(self.output_grid.grid)

    @property
    def output_cols(self):
        return len(self.output_grid.grid[0]) if self.output_grid.grid else 0

    def pad_example(
        self, tokenizer: PreTrainedTokenizer, max_rows: int, max_cols: int
    ) -> tuple:
        """Pad the example to the specified size.

        Args:
            tokenizer: PreTrainedTokenizer for tokenization
            max_rows: maximum number of rows
            max_cols: maximum number of columns
            mask: whether to mask the grid (default: False)
        Returns:
            padded_example: padded example grid
            padded_example_ids: padded grid filled with example_id
            padded_example_grid_types: padded grid filled with grid_type
        """
        padded_input_grid, padded_input_example_ids, padded_input_grid_types = \
            self.input_grid.pad_grid(tokenizer, max_rows, max_cols, mask=False)
        if self.example_type == 1:  # mask the test output grid
            padded_output_grid, padded_output_example_ids, padded_output_grid_types = \
                self.output_grid.pad_grid(tokenizer, max_rows, max_cols, mask=True)

        grid_boundary = np.array([tokenizer.arc_sep_grid_token] * padded_input_grid.shape[0])
        padded_example = np.concatenate((padded_input_grid, grid_boundary[:, None], padded_output_grid), axis=1)

        example_ids_boundary = np.full((padded_input_example_ids.shape[0], 1), self.example_id)
        grid_types_boundary = np.full((padded_input_grid_types.shape[0], 1), self.input_grid.grid_type)

        padded_example_ids = np.concatenate(
            (padded_input_example_ids, example_ids_boundary, padded_output_example_ids),
            axis=1
        )
        padded_example_grid_types = np.concatenate(
            (padded_input_grid_types, grid_types_boundary, padded_output_grid_types),
            axis=1
        )

        assert padded_example.shape == padded_example_ids.shape, \
            f"Shape mismatch: padded_example.shape = {padded_example.shape}" + \
            f"vs padded_example_ids.shape = {padded_example_ids.shape}"
        assert padded_example.shape == padded_example_grid_types.shape, \
            f"Shape mismatch: padded_example.shape = {padded_example.shape}" + \
            f"vs padded_example_grid_types.shape = {padded_example_grid_types.shape}"

        return (padded_example, padded_example_ids, padded_example_grid_types)

    def unpad_example(self, padded_example: List[List[Any]], tokenizer: PreTrainedTokenizer) -> List[List[int]]:
        """Unpad the example to its original size.

        Args:
            padded_example: padded example to unpad
        Returns:
            unpadded_example: unpadded example with the original size
        """
        unpadded_example = []
        special_tokens = list(tokenizer.special_tokens_maps.keys())

        for row in padded_example:
            if row[0] == tokenizer.arc_pad_token:
                continue
            unpadded_row = [int(cell) for cell in row if cell not in special_tokens]
            unpadded_example.append(unpadded_row)

        # Validate that grid contains only integers
        if not all(isinstance(cell, int) for row in unpadded_example for cell in row):
            raise ValueError("Unpadded grid must contain only integers")
        # Validate that all rows have the same length
        if not all(len(row) == len(unpadded_example[0]) for row in unpadded_example):
            raise ValueError("All rows in the unpadded grid must have the same length")
        # Validate that grid is a 2D list
        if not unpadded_example or not isinstance(unpadded_example[0], list):
            raise ValueError("Unpadded grid must be a 2D list")

        return unpadded_example


@dataclass
class ARCTask:
    """Class for storing an ARC task with multiple ARCExamples."""
    task_id: str
    config: dict
    examples: List[ARCExample] = field(default_factory=list)  # Initialize to an empty list

    def __post_init__(self):
        train_examples = self.config.get("train", [])
        test_examples = self.config.get("test", [])
        assert len(test_examples) != 0, "Test example is required"

        self.examples.clear()  # Clear any existing examples

        for idx, example in enumerate(train_examples):
            self.examples.append(
                ARCExample(
                    task_id=self.task_id,
                    input_grid=ARCGrid(grid=example["input"], example_id=idx, grid_type=0),
                    output_grid=ARCGrid(grid=example["output"], example_id=idx, grid_type=1),
                    example_id=idx,
                    example_type=0  # 0 for train
                )
            )

        for idx, example in enumerate(test_examples):
            self.examples.append(
                ARCExample(
                    task_id=self.task_id,
                    input_grid=ARCGrid(grid=example["input"], example_id=idx, grid_type=0),
                    output_grid=ARCGrid(grid=example["output"], example_id=idx, grid_type=1),
                    example_id=idx + len(train_examples),
                    example_type=1  # 1 for test
                )
            )

    @property
    def num_examples(self):
        return len(self.examples)


class ARCMaskedDataset(Dataset):
    """Dataset for ARC examples with support for object encoding"""
    def __init__(self, tasks, tokenizer, max_length=20480, extract_objects=False):
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extract_objects = extract_objects

        # Get token IDs for special tokens
        self.start_grid_token_id = tokenizer.convert_tokens_to_ids("<start_grid>")
        self.end_grid_token_id = tokenizer.convert_tokens_to_ids("<end_grid>")
        self.start_row_token_id = tokenizer.convert_tokens_to_ids("<start_row>")
        self.end_row_token_id = tokenizer.convert_tokens_to_ids("<end_row>")

    def __len__(self):
        return len(self.tasks)

    def extract_grid_objects(self, grid):
        """
        Extract objects from a grid using Connected Component Labeling algorithm.
        TODO: use openCV's implementation to improve performance.

        Args:
            grid: 2D grid of values

        Returns:
            object_map: Same shape as grid, containing object IDs
        """
        rows, cols = len(grid), len(grid[0])
        visited = set()
        object_map = [[0 for _ in range(cols)] for _ in range(rows)]
        object_id = 1

        def get_neighbors(r, c):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity
            neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == grid[r][c]:
                    neighbors.append((nr, nc))
            return neighbors

        def dfs(r, c, obj_id):
            stack = [(r, c)]
            color = grid[r][c]

            while stack:
                curr_r, curr_c = stack.pop()
                if (curr_r, curr_c) in visited:
                    continue

                visited.add((curr_r, curr_c))
                object_map[curr_r][curr_c] = obj_id

                for nr, nc in get_neighbors(curr_r, curr_c):
                    if (nr, nc) not in visited and grid[nr][nc] == color:
                        stack.append((nr, nc))

        # Find connected components
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid[r][c] != 0:  # Skip background (0 == background)
                    dfs(r, c, object_id)
                    object_id += 1

        return object_map

    def __getitem__(self, idx):
        """
        Get tokenized and encoded example with grid position information.

        Returns:
            Dictionary with:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - grid_ids: Grid ID for each token
                - grid_types: Grid type (0=input, 1=output) for each token
                - row_positions: Row position for each token
                - col_positions: Column position for each token
                - object_idx: Object indices for each token (if extract_objects=True)
                - tgt_ids: Target IDs
        """
        example = self.examples[idx]

        # Initialize sequence data
        token_ids = []
        grid_ids = []
        grid_types = []
        row_positions = []
        col_positions = []
        object_idx = [] if self.extract_objects else None

        # Extract objects from grids if needed
        input_objects = None
        output_objects = None
        if self.extract_objects:
            input_objects = self.extract_grid_objects(example.input_grid)
            output_objects = self.extract_grid_objects(example.output_grid)

        # Function to add a grid to the sequence
        def add_grid(grid, grid_id, grid_type, object_grid=None):
            # Add start grid token
            token_ids.append(self.start_grid_token_id)
            grid_ids.append(grid_id)
            grid_types.append(grid_type)
            row_positions.append(0)  # Special position for grid tokens
            col_positions.append(0)  # Special position for grid tokens
            if self.extract_objects:
                object_idx.append(0)  # Special object ID for grid tokens

            # Add each row
            for row_idx, row in enumerate(grid):
                # Add start row token
                token_ids.append(self.start_row_token_id)
                grid_ids.append(grid_id)
                grid_types.append(grid_type)
                row_positions.append(row_idx)
                col_positions.append(-1)  # Special position for row tokens
                if self.extract_objects:
                    object_idx.append(0)  # Special object ID for row tokens

                # Add each cell in the row
                for col_idx, cell_value in enumerate(row):
                    # Convert cell value to token ID (assuming 0-9 values map to token IDs 0-9)
                    token_ids.append(cell_value)
                    grid_ids.append(grid_id)
                    grid_types.append(grid_type)
                    row_positions.append(row_idx)
                    col_positions.append(col_idx)
                    if self.extract_objects and object_grid:
                        object_idx.append(object_grid[row_idx][col_idx])
                    elif self.extract_objects:
                        object_idx.append(0)

                # Add end row token
                token_ids.append(self.end_row_token_id)
                grid_ids.append(grid_id)
                grid_types.append(grid_type)
                row_positions.append(row_idx)
                col_positions.append(len(row))  # Special position for row tokens
                if self.extract_objects:
                    object_idx.append(0)  # Special object ID for row tokens

            # Add end grid token
            token_ids.append(self.end_grid_token_id)
            grid_ids.append(grid_id)
            grid_types.append(grid_type)
            row_positions.append(len(grid))  # Special position for grid tokens
            col_positions.append(0)  # Special position for grid tokens
            if self.extract_objects:
                object_idx.append(0)  # Special object ID for grid tokens

        # Add input grid
        add_grid(example.input_grid, example.grid_id, 0, input_objects)  # 0 = input grid

        # Add output grid
        add_grid(example.output_grid, example.grid_id, 1, output_objects)  # 1 = output grid

        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        grid_ids = torch.tensor(grid_ids, dtype=torch.long)
        grid_types = torch.tensor(grid_types, dtype=torch.long)
        row_positions = torch.tensor(row_positions, dtype=torch.long)
        col_positions = torch.tensor(col_positions, dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'grid_ids': grid_ids,
            'grid_types': grid_types,
            'row_positions': row_positions,
            'col_positions': col_positions
        }

        if self.extract_objects:
            result['object_idx'] = torch.tensor(object_idx, dtype=torch.long)

        return result
