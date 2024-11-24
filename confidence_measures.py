
import torch
import torch.nn.functional as F
from typing import Callable

def softmax_diff(logits: torch.Tensor) -> torch.Tensor:
    """
    Softmax top 2 prob diff。
    Input: logits (torch.Tensor): [batch_size, vocab_size]
    Returns: torch.Tensor: [batch_size]
    """
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    return sorted_probs[:, 0] - sorted_probs[:, 1]



def softmax_max(logits: torch.Tensor) -> torch.Tensor:
    """
    Softmax top 1 prob。
    Input: logits (torch.Tensor): [batch_size, vocab_size]
    Returns:torch.Tensor: [batch_size]
    """
    probs = F.softmax(logits, dim=-1)
    return torch.max(probs, dim=-1).values



def state_cosine_similarity(prev_state: torch.Tensor, new_state: torch.Tensor) -> torch.Tensor:
    """
    cosine similarity of two hidden layers。
    Input:
        prev_state (torch.Tensor): [batch_size, hidden_size]
        new_state (torch.Tensor): [batch_size, hidden_size]
    Returns:
        torch.Tensor: [batch_size]
    """
    numerator = torch.sum(prev_state * new_state, dim=-1)
    denominator = torch.norm(prev_state, dim=-1) * torch.norm(new_state, dim=-1) + 1e-8
    return numerator / denominator



def compute_confidence(
    logits: torch.Tensor,
    prev_state: torch.Tensor,
    new_state: torch.Tensor,
    conf_method: str = 'softmax_max'
) -> torch.Tensor:
    """
    Args:
        logits (torch.Tensor): [batch_size, vocab_size]
        prev_state (torch.Tensor): [batch_size, hidden_size]
        new_state (torch.Tensor): [batch_size, hidden_size]
        conf_method (str): 。

    Returns:
        torch.Tensor: [batch_size]
    """

    if conf_method == 'softmax_diff':
        return softmax_diff(logits)
    elif conf_method == 'softmax_max':
        return softmax_max(logits)
    elif conf_method == 'state':
        return state_cosine_similarity(prev_state, new_state)
    else:
        raise ValueError(f"Unknown confidence method: {conf_method}")


def should_exit(
    confidence: torch.Tensor,
    threshold: float
) -> torch.Tensor:
    """
    calculated: confidence ; and parameter threshold。
    Args:
        confidence (torch.Tensor): [batch_size]
        threshold (float)
    Returns:
        torch.Tensor: [batch_size] bool value ，True for exit 。
    """
    return confidence >= threshold