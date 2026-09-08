__all__ = ["CausalDiffusionInferencePipeline", "CausalInferencePipeline"]


def __getattr__(name):
    if name in __all__:
        from .pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline

        return {
            "CausalDiffusionInferencePipeline": CausalDiffusionInferencePipeline,
            "CausalInferencePipeline": CausalInferencePipeline,
        }[name]
    raise AttributeError(name)
