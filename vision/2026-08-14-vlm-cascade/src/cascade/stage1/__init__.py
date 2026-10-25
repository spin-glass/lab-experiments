"""cascade.stage1 — Stage1（エンコーダ zero-shot。design.md §1、taxonomy.md §5）。"""

from cascade.stage1.zeroshot import ZeroShot, build_zeroshot

__all__ = ["ZeroShot", "build_zeroshot"]
