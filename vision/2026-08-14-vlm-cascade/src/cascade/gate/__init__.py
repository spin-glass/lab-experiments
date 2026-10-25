"""cascade.gate — 非料理画像の距離信号ゲート G0（g0_nonfood_plan.md、design.md §6 G0）。

- ``threshold``: 動作点（tune_cal の正しく food の 1% 位置、副 0.5% / 2%）の決定と適用（§7.1）
- ``splits``: 役割ガード（eval を fit・閾値決定に渡すと ``SplitLeakError``）、LOTO / LOSO fold（§3.1、§5.4）
- ``scorers``: 検出器（MSP / kNN / Mahalanobis++ / C 系。大 = 対象内、fit 前 score は raise）（§5）
- ``pipeline`` / ``data_loader``: 主 8 構成の実行・記述分析・帰結表（§6〜§8）

サブモジュールは重い依存を持ち得るため、ここでは軽量な threshold / splits のみ再エクスポートする。
"""

from cascade.gate.splits import (
    Fold,
    RoleTable,
    SplitGuard,
    SplitLeakError,
    assert_no_eval,
    fold_evaluable,
    loso_folds,
    loto_folds,
    summarize_roles,
    type_source_crosstab,
)
from cascade.gate.threshold import (
    LOSSES_DEFAULT,
    ThresholdResult,
    apply_threshold,
    fit_threshold,
    fit_thresholds,
    k_index_for,
)

__all__ = [
    "LOSSES_DEFAULT",
    "Fold",
    "RoleTable",
    "SplitGuard",
    "SplitLeakError",
    "ThresholdResult",
    "apply_threshold",
    "assert_no_eval",
    "fit_threshold",
    "fit_thresholds",
    "fold_evaluable",
    "k_index_for",
    "loso_folds",
    "loto_folds",
    "summarize_roles",
    "type_source_crosstab",
]
