"""Detectores usados pelo pipeline ``experiments/`` (cópia sem alteração; ver NOTICE.md)."""

from fault_diagnosis.technique.base_detector import BaseDetector
from fault_diagnosis.technique.autoencoder import MWAutoEncoder
from fault_diagnosis.technique.rmwpcasc import RMWPCA_SC

__all__ = ["BaseDetector", "MWAutoEncoder", "RMWPCA_SC"]
