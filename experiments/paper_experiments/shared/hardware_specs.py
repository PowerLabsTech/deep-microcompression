"""
Hardware target specifications and NAS filter/condition factory functions.

Absolute byte budgets are used instead of ratios relative to the original model,
so filters are hardware-grounded rather than model-relative.
"""

# ---------------------------------------------------------------------------
# Board definitions
# ---------------------------------------------------------------------------

ATMEGA328P = {
    "name": "ATmega328P (Arduino Uno)",
    "sram_bytes": 2 * 1024,        # 2 KB
    "flash_bytes": 32 * 1024,      # 32 KB
    # Leave ~200 B headroom for stack + drivers
    "workspace_budget": 1_800,
    # Leave ~12 KB headroom for compiled inference code
    "weight_budget": 20_000,
}

ATMEGA2560 = {
    "name": "ATmega2560 (Arduino Mega)",
    "sram_bytes": 8 * 1024,        # 8 KB
    "flash_bytes": 256 * 1024,     # 256 KB
    # Leave ~2 KB headroom for stack + globals
    "workspace_budget": 6_144,
    # Leave ~56 KB headroom for inference code
    "weight_budget": 200_000,
}

# ---------------------------------------------------------------------------
# Filter / condition factories
# ---------------------------------------------------------------------------

def make_nas_filter(hw: dict, input_shape: tuple):
    """
    Returns a filter callable for get_nas_compression_data().
    Rejects configs that would exceed the board's workspace or weight budget,
    or that don't use static quantization (required for integer-only inference).
    """
    from development import QuantizationScheme, Sequential

    ws_budget = hw["workspace_budget"]
    wt_budget = hw["weight_budget"]

    def nas_filter(compressed_model: Sequential, compression_config: dict) -> bool:
        if compression_config["quantize"]["scheme"] != QuantizationScheme.STATIC:
            return False
        if compressed_model.get_workspace_size(input_shape) > ws_budget:
            return False
        if compressed_model.get_size_in_bytes() > wt_budget:
            return False
        return True

    return nas_filter


def make_ultra_condition(hw: dict):
    """
    Condition for DMC-Ultra search: fits on the board, no accuracy floor.
    The evolutionary search then minimises weight bytes.
    """
    from development import QuantizationScheme

    ws_budget = hw["workspace_budget"]
    wt_budget = hw["weight_budget"]

    def condition(metric: float, size: int, ram: int, config: dict) -> bool:
        if config["quantize"]["scheme"] != QuantizationScheme.STATIC:
            return False
        if ram > ws_budget:
            return False
        if size > wt_budget:
            return False
        return True

    return condition


def make_board_condition(hw: dict, baseline_accuracy: float, max_drop: float = 5.0):
    """
    Condition for DMC-Board search: fits on the board AND accuracy drop is bounded.
    The evolutionary search then maximises accuracy.

    Args:
        hw: hardware spec dict.
        baseline_accuracy: top-1 % of the uncompressed model.
        max_drop: maximum allowable accuracy drop in percentage points.
    """
    from development import QuantizationScheme

    ws_budget = hw["workspace_budget"]
    wt_budget = hw["weight_budget"]

    def condition(metric: float, size: int, ram: int, config: dict) -> bool:
        if config["quantize"]["scheme"] != QuantizationScheme.STATIC:
            return False
        if ram > ws_budget:
            return False
        if size > wt_budget:
            return False
        if baseline_accuracy - metric > max_drop:
            return False
        return True

    return condition
