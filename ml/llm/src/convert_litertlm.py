"""Milestone 7: convert the merged fine-tuned checkpoint to .litertlm via
litert-torch's export_hf (docs/PRODUCT_SPEC.md section 15 — "official AI Edge
Torch/LiteRT-LM conversion flow"; ai-edge-torch itself is deprecated in favor
of litert-torch, the current package name).

Applies one targeted monkey-patch before calling export(): the installed
litert-torch==0.9.1 (the latest available) is missing a concrete
implementation of `get_max_length` on `LiteRTLMCacheLayer`, which is an
abstract method required by the installed transformers version's cache base
class — a genuine version-skew bug between two Google/HF libraries, not
something introduced here. Confirmed by direct inspection: the class already
computes `self.max_cache_len` in `__init__`, exactly matching the fixed-size
StaticLayer implementation in transformers/cache_utils.py, which returns
`self.max_cache_len` from its own `get_max_length` (other cache types there
return different things appropriate to their semantics, e.g. -1 for dynamic/
unbounded caches, `self.sliding_window` for sliding-window caches — checked
each, not assumed uniform).
"""

import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"


def apply_get_max_length_patch() -> None:
    from litert_torch.generative.export_hf.core import cache as cache_module

    target_cls = cache_module.LiteRTLMCacheLayer
    abstract_names = getattr(target_cls, "__abstractmethods__", frozenset())
    if "get_max_length" not in abstract_names:
        return  # a future litert-torch release may fix this upstream; don't double-patch

    # hasattr() alone is NOT a reliable "already patched" check here: an
    # unimplemented @abstractmethod is still a real attribute findable via
    # the MRO, so hasattr() returns True even though the class can't be
    # instantiated (confirmed the hard way: an earlier version of this patch
    # used a hasattr() guard that silently skipped the actual assignment
    # every time, then wondered why the original TypeError kept recurring).
    def get_max_length(self) -> int:
        return self.max_cache_len

    target_cls.get_max_length = get_max_length
    # ABCMeta computes __abstractmethods__ once at class-creation time and
    # does not recompute it when a method is assigned after the fact —
    # instantiation stays blocked unless the name is removed explicitly.
    target_cls.__abstractmethods__ = frozenset(abstract_names - {"get_max_length"})
    print("Applied get_max_length monkey-patch to LiteRTLMCacheLayer (upstream litert-torch/transformers version skew).")


def main() -> int:
    import yaml

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = (base_dir / config["paths"]["artifacts_dir"]).resolve()
    merged_checkpoint = artifacts_dir / "merged_checkpoint"
    output_dir = artifacts_dir / "litertlm_output"

    if not merged_checkpoint.exists():
        print(f"ERROR: {merged_checkpoint} not found. Run merge_adapter.py first.")
        return 1

    apply_get_max_length_patch()

    from litert_torch.generative.export_hf import export

    output_dir.mkdir(parents=True, exist_ok=True)
    export.export(
        model=str(merged_checkpoint),
        output_dir=str(output_dir),
        bundle_litert_lm=True,
    )
    print(f"Conversion complete. Output in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
