-- D14: 誤混入削減率 r / n_before（docs/g0_nonfood_plan.md §7.2、design.md §8 D14）
-- 層 A eval（operational_truth = out_of_scope、主 5 型）で、P0 が food 確定した枚数 n_before と
-- そのうちゲートが除去した枚数 r を、構成（mode × method_id × threshold_id）× subject_type で数える。
-- 'ALL' 行が合算（主検定の入力）。DuckDB / BigQuery 両対応（FILTER 句・QUALIFY は使わない）。
-- プレースホルダ（識別子）: {{gate_decisions}} {{scope_annotations}}。リテラル: $run_id（run_sql(..., literals={"run_id": ...})。BigQuery では @run_id）
WITH layer_a AS (
  SELECT photo_id, subject_type
  FROM {{scope_annotations}}
  WHERE operational_truth = 'out_of_scope'
    AND role = 'eval'
    AND subject_type IN (
      'doll_character_statue', 'amusement_tourist', 'signage_decor_goods',
      'person_animal', 'vehicle_street'
    )
),
gated AS (
  SELECT g.mode, g.method_id, g.threshold_id, a.subject_type,
         CASE WHEN g.stage1_pred = 'food' THEN 1 ELSE 0 END AS is_before,
         CASE WHEN g.stage1_pred = 'food' AND g.decision = 'reject' THEN 1 ELSE 0 END AS is_removed
  FROM {{gate_decisions}} AS g
  JOIN layer_a AS a ON a.photo_id = g.photo_id
  WHERE g.run_id = $run_id
    AND g.mode IN ('P1', 'P2')
),
by_type AS (
  SELECT mode, method_id, threshold_id, subject_type,
         SUM(is_before) AS n_before, SUM(is_removed) AS r
  FROM gated
  GROUP BY mode, method_id, threshold_id, subject_type
),
pooled AS (
  SELECT mode, method_id, threshold_id, 'ALL' AS subject_type,
         SUM(is_before) AS n_before, SUM(is_removed) AS r
  FROM gated
  GROUP BY mode, method_id, threshold_id
)
SELECT mode, method_id, threshold_id, subject_type, n_before, r,
       CASE WHEN n_before > 0 THEN CAST(r AS DOUBLE) / CAST(n_before AS DOUBLE) ELSE NULL END AS reduction
FROM (
  SELECT * FROM pooled
  UNION ALL
  SELECT * FROM by_type
) AS u
ORDER BY mode, method_id, threshold_id, CASE WHEN subject_type = 'ALL' THEN 0 ELSE 1 END, subject_type
