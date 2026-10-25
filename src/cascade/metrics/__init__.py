"""G0 の統計指標（g0_nonfood_plan.md §6〜§8）。二項 CI・多重比較・対応比較・比率・ブートストラップ・記述分析。"""

from cascade.metrics.binom import binom_test_one_sided_greater, clopper_pearson, wilson
from cascade.metrics.bootstrap import (
    cluster_bootstrap,
    delta_method_ratio,
    food_loss_confirmed_bootstrap,
)
from cascade.metrics.descriptive import (
    auroc,
    fpr_at_tpr,
    histogram_table,
    kl_divergence_hist,
    overlap_coefficient,
    quadrant_counts,
)
from cascade.metrics.holm import holm
from cascade.metrics.mcnemar import mcnemar_exact
from cascade.metrics.rates import (
    Rate,
    false_acceptance,
    food_loss_confirmed,
    food_loss_nominal,
    other_class_frr,
    pass_rate,
    rate,
    reduction,
)

__all__ = [
    "Rate",
    "auroc",
    "binom_test_one_sided_greater",
    "clopper_pearson",
    "cluster_bootstrap",
    "delta_method_ratio",
    "false_acceptance",
    "food_loss_confirmed",
    "food_loss_confirmed_bootstrap",
    "food_loss_nominal",
    "fpr_at_tpr",
    "histogram_table",
    "holm",
    "kl_divergence_hist",
    "mcnemar_exact",
    "other_class_frr",
    "overlap_coefficient",
    "pass_rate",
    "quadrant_counts",
    "rate",
    "reduction",
    "wilson",
]
