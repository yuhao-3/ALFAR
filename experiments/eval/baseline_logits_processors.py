"""
Logits Processors for Baseline Methods

This module implements various logits processing strategies for baseline methods:
- CD: Contrastive Decoding
- CAD: Context-Aware Decoding
- AdaCAD: Adaptive Context-Aware Decoding
- Entropy-based Decoding
- COIECD: Contextual Information-Entropy Constraint Decoding
"""

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessor
import numpy as np


class ContrastiveDecodingLogitsProcessor(LogitsProcessor):
    """
    Contrastive Decoding (CD)

    Contrasts the likelihood under a large LM (expert) and a small LM (amateur).
    CD(x) = log P_expert(x) - alpha * log P_amateur(x)

    Reference: "Contrastive Decoding: Open-ended Text Generation as Optimization" (ACL 2023)
    """

    def __init__(self, alpha=0.5, cutoff=0.1):
        """
        Args:
            alpha: Weight for amateur model logits
            cutoff: Plausibility constraint threshold
        """
        self.alpha = alpha
        self.cutoff = cutoff
        self.amateur_logits = None

    def set_amateur_logits(self, amateur_logits):
        """Set logits from amateur model"""
        self.amateur_logits = amateur_logits

    def __call__(self, input_ids, scores):
        if self.amateur_logits is None:
            return scores

        # Apply contrastive decoding
        # scores are expert logits (log probabilities)
        # Subtract weighted amateur logits
        contrast_scores = scores - self.alpha * self.amateur_logits

        # Apply plausibility constraint (mask out low probability tokens from expert)
        expert_probs = F.softmax(scores, dim=-1)
        mask = expert_probs < self.cutoff
        contrast_scores[mask] = -float('inf')

        return contrast_scores


class ContextAwareDecodingLogitsProcessor(LogitsProcessor):
    """
    Context-Aware Decoding (CAD)

    Amplifies the difference between output probabilities when model is used
    with and without context.

    CAD(x) = (1 + alpha) * log P_context(x) - alpha * log P_no_context(x)

    Reference: "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding" (NAACL 2024)
    """

    def __init__(self, alpha=0.5):
        """
        Args:
            alpha: Weight for amplifying context effect
        """
        self.alpha = alpha
        self.no_context_logits = None

    def set_no_context_logits(self, no_context_logits):
        """Set logits from no-context model"""
        self.no_context_logits = no_context_logits

    def __call__(self, input_ids, scores):
        if self.no_context_logits is None:
            return scores

        # Apply context-aware decoding
        # scores are context-based logits
        cad_scores = (1 + self.alpha) * scores - self.alpha * self.no_context_logits

        return cad_scores


class AdaptiveCADLogitsProcessor(LogitsProcessor):
    """
    Adaptive Context-Aware Decoding (AdaCAD)

    Dynamically adjusts the weight of contextual knowledge based on the
    degree of conflict between parametric and contextual knowledge.

    Reference: "AdaCAD: Adaptively Decoding to Balance Conflicts between
                Contextual and Parametric Knowledge" (2024)
    """

    def __init__(self, alpha_max=1.0, threshold=0.5):
        """
        Args:
            alpha_max: Maximum weight for context amplification
            threshold: Threshold for detecting conflict
        """
        self.alpha_max = alpha_max
        self.threshold = threshold
        self.no_context_logits = None

    def set_no_context_logits(self, no_context_logits):
        """Set logits from no-context model"""
        self.no_context_logits = no_context_logits

    def compute_conflict_degree(self, context_logits, no_context_logits):
        """
        Compute degree of conflict between context and no-context distributions.
        Uses KL divergence as measure of conflict.
        """
        context_probs = F.softmax(context_logits, dim=-1)
        no_context_probs = F.softmax(no_context_logits, dim=-1)

        # KL divergence: KL(P_context || P_no_context)
        kl_div = F.kl_div(no_context_probs.log(), context_probs, reduction='sum')

        # Normalize to [0, 1] range (using tanh to bound)
        conflict_degree = torch.tanh(kl_div / 10.0)  # Scale factor 10.0 is empirical

        return conflict_degree.item()

    def __call__(self, input_ids, scores):
        if self.no_context_logits is None:
            return scores

        # Compute adaptive alpha based on conflict degree
        conflict_degree = self.compute_conflict_degree(scores, self.no_context_logits)

        # Adaptive weight: higher conflict → higher alpha
        if conflict_degree > self.threshold:
            alpha = self.alpha_max * ((conflict_degree - self.threshold) / (1.0 - self.threshold))
        else:
            alpha = 0.0

        # Apply adaptive CAD
        adacad_scores = (1 + alpha) * scores - alpha * self.no_context_logits

        return adacad_scores


class EntropyBasedDecodingLogitsProcessor(LogitsProcessor):
    """
    Entropy-Based Decoding

    Uses entropy to guide decoding, preferring low-entropy (confident) predictions.
    """

    def __init__(self, entropy_threshold=1.0, beta=0.5):
        """
        Args:
            entropy_threshold: Threshold for high entropy
            beta: Weight for entropy penalty
        """
        self.entropy_threshold = entropy_threshold
        self.beta = beta

    def compute_entropy(self, logits):
        """Compute entropy of probability distribution"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def __call__(self, input_ids, scores):
        # Compute entropy
        entropy = self.compute_entropy(scores)

        # If entropy is high (model is uncertain), apply penalty
        if entropy > self.entropy_threshold:
            # Sharpen distribution to reduce uncertainty
            scores = scores / (1.0 + self.beta * (entropy - self.entropy_threshold))

        return scores


class COIECDLogitsProcessor(LogitsProcessor):
    """
    Contextual Information-Entropy Constraint Decoding (COIECD)

    Combines contextual information with entropy constraints to guide decoding.
    """

    def __init__(self, alpha=0.5, entropy_threshold=1.0, beta=0.5):
        """
        Args:
            alpha: Weight for context amplification
            entropy_threshold: Threshold for high entropy
            beta: Weight for entropy penalty
        """
        self.alpha = alpha
        self.entropy_threshold = entropy_threshold
        self.beta = beta
        self.no_context_logits = None

    def set_no_context_logits(self, no_context_logits):
        """Set logits from no-context model"""
        self.no_context_logits = no_context_logits

    def compute_entropy(self, logits):
        """Compute entropy of probability distribution"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def __call__(self, input_ids, scores):
        if self.no_context_logits is None:
            return scores

        # Apply CAD component
        cad_scores = (1 + self.alpha) * scores - self.alpha * self.no_context_logits

        # Compute entropy
        entropy = self.compute_entropy(cad_scores)

        # Apply entropy constraint
        if entropy > self.entropy_threshold:
            # Sharpen distribution to reduce uncertainty
            cad_scores = cad_scores / (1.0 + self.beta * (entropy - self.entropy_threshold))

        return cad_scores


class VCDLogitsProcessor(LogitsProcessor):
    """
    Visual Contrastive Decoding (VCD)

    Contrasts output distributions from original and distorted visual inputs.
    """

    def __init__(self, alpha=0.5):
        """
        Args:
            alpha: Weight for distorted logits
        """
        self.alpha = alpha
        self.distorted_logits = None

    def set_distorted_logits(self, distorted_logits):
        """Set logits from distorted image input"""
        self.distorted_logits = distorted_logits

    def __call__(self, input_ids, scores):
        if self.distorted_logits is None:
            return scores

        # Apply visual contrastive decoding
        vcd_scores = scores - self.alpha * self.distorted_logits

        return vcd_scores
