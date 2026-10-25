-- D14: 料理損失（名目）（docs/g0_nonfood_plan.md §7.2）
-- eval の「正しく food」（Yelp ラベル food ∧ P0 = food）のうち、ゲートで棄却された率を構成ごとに数える。
-- 確認済み損失（目視 q̂）は SQL では出さない（review jsonl はパイプライン側で結合する）。
-- Yelp のラベルは {{labels}}（label_source = 'yelp'）。gate_decisions には stage の集合しか無いので役割で絞らない。
-- プレースホルダ（識別子）: {{gate_decisions}} {{labels}}。リテラル: $run_id（run_sql(..., literals={"run_id": ...})。BigQuery では @run_id）
WITH yelp_food AS (
  SELECT photo_id
  FROM {{labels}}
  WHERE label_source = 'yelp' AND label = 'food'
),
correct_food AS (
  SELECT g.mode, g.method_id, g.threshold_id, g.photo_id,
         CASE WHEN g.decision = 'reject' THEN 1 ELSE 0 END AS is_rejected
  FROM {{gate_decisions}} AS g
  JOIN yelp_food AS y ON y.photo_id = g.photo_id
  WHERE g.run_id = $run_id
    AND g.mode IN ('P1', 'P2')
    AND g.stage1_pred = 'food'
)
SELECT mode, method_id, threshold_id,
       COUNT(*) AS n_correct_food,
       SUM(is_rejected) AS n_rejected,
       CAST(SUM(is_rejected) AS DOUBLE) / CAST(COUNT(*) AS DOUBLE) AS food_loss_nominal
FROM correct_food
GROUP BY mode, method_id, threshold_id
ORDER BY mode, method_id, threshold_id
