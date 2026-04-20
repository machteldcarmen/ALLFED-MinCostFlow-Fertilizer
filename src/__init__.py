"""Min-cost-flow model of global fertilizer trade.

Main interface::

    from src.model import FertilizerMCF
    model = FertilizerMCF(supply, demand, edges)
    result = model.solve()
    print(result.flow_matrix)
"""

from .model import FertilizerMCF, MCFResult
from .utils import use_allfed_style

__all__ = ["FertilizerMCF", "MCFResult", "use_allfed_style"]
