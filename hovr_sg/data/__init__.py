from .schema import ObjectAnnotation, RelationAnnotation, SceneRecord

try:
    from .unified_dataset import UnifiedSceneGraphDataset, collate_scene_graph
    from .samplers import RepeatFactorSampler
except (ImportError, ModuleNotFoundError):  # Optional until torch/torchvision are installed.
    UnifiedSceneGraphDataset = None
    collate_scene_graph = None
    RepeatFactorSampler = None

__all__ = ["ObjectAnnotation", "RelationAnnotation", "SceneRecord", "UnifiedSceneGraphDataset", "collate_scene_graph", "RepeatFactorSampler"]
