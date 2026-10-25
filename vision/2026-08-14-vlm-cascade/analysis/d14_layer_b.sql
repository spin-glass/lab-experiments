-- D14: 層 B（対象内の非料理）の P1 除去率と P2 通過率（docs/g0_nonfood_plan.md §7.2）
-- 層 B = content ∈ {non_food, mixed} ∧ operational_truth ∈ {inside, outside, menu}。
-- P1: P0 が food と判定した層 B のうち P1 が food から外した率（高いほど良い）。
-- P2: 層 B のうち P2 を通過した率（片側 95% 下限で判定。CI はパイプライン側）。
-- プレースホルダ（識別子）: {{gate_decisions}} {{scope_annotations}}。リテラル: $run_id（run_sql(..., literals={"run_id": ...})。BigQuery では @run_id）
WITH layer_b AS (
  SELECT photo_id
  FROM {{scope_annotations}}
  WHERE content IN ('non_food', 'mixed')
    AND operational_truth IN ('inside', 'outside', 'menu')
),
gated AS (
  SELECT g.mode, g.method_id, g.threshold_id,
         CASE WHEN g.stage1_pred = 'food' THEN 1 ELSE 0 END AS p0_food,
         CASE WHEN g.stage1_pred = 'food' AND g.decision = 'reject' THEN 1 ELSE 0 END AS p1_removed,
         CASE WHEN g.decision <> 'reject' THEN 1 ELSE 0 END AS passed
  FROM {{gate_decisions}} AS g
  JOIN layer_b AS b ON b.photo_id = g.photo_id
  WHERE g.run_id = $run_id
    AND g.mode IN ('P1', 'P2')
)
SELECT mode, method_id, threshold_id,
       COUNT(*) AS n_layer_b,
       SUM(p0_food) AS n_p0_food,
       SUM(p1_removed) AS n_removed_from_food,
       CASE WHEN mode = 'P1' AND SUM(p0_food) > 0
            THEN CAST(SUM(p1_removed) AS DOUBLE) / CAST(SUM(p0_food) AS DOUBLE) ELSE NULL END AS p1_removal_rate,
       CASE WHEN mode = 'P2' AND COUNT(*) > 0
            THEN CAST(SUM(passed) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE) ELSE NULL END AS p2_pass_rate
FROM gated
GROUP BY mode, method_id, threshold_id
ORDER BY mode, method_id, threshold_id
