"""Simple registry system for managing models, datasets, etc."""

from typing import Type, Dict, Callable, Any

class Registry:
    """A simple class-based registry.

    Example usage:
        my_registry = Registry("models")

        @my_registry.register('resnet')
        class ResNet:
            pass

        # Get the class
        model_class = my_registry.get('resnet')
        # Instantiate
        model_instance = model_class()
    """
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        """Class decorator to register a class under a given name."""
        if not isinstance(name, str):
            raise TypeError(f"Registry key must be a string, got {type(name)}")
        if name in self._registry:
            print(f"Warning: Overwriting registry key '{name}' in registry '{self._name}'")

        def decorator(cls: Type) -> Type:
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type:
        """Retrieve an item from the registry by name."""
        if name not in self._registry:
            raise KeyError(f"Key '{name}' not found in registry '{self._name}'. Available keys: {list(self._registry.keys())}")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name='{self._name}', keys={list(self._registry.keys())})"

    def items(self):
        return self._registry.items()

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()


# --- Define specific registries used in the project ---

FEATURE_EXTRACTOR_REGISTRY = Registry("feature_extractors")
FEATURE_UPSAMPLER_REGISTRY = Registry("feature_upsamplers")
DATASET_REGISTRY = Registry("datasets")
PIPELINE_REGISTRY = Registry("pipelines")
REGISTRATION_ESTIMATOR_REGISTRY = Registry("registration_estimators")

# You can add more registries as needed


if __name__ == '__main__':
    print("Testing Registry System...")

    # Example Usage
    MODEL_REGISTRY = Registry("test_models")

    @MODEL_REGISTRY.register('model_a')
    class ModelA:
        def __init__(self, param=1):
            self.param = param
            print(f"ModelA initialized with param={self.param}")

    @MODEL_REGISTRY.register('model_b')
    class ModelB:
        def __init__(self, name='b'):
            self.name = name
            print(f"ModelB initialized with name={self.name}")

    print(MODEL_REGISTRY)
    assert 'model_a' in MODEL_REGISTRY
    assert len(MODEL_REGISTRY) == 2

    # Get class and instantiate
    RetrievedModelA = MODEL_REGISTRY.get('model_a')
    instance_a = RetrievedModelA(param=10)
    assert isinstance(instance_a, ModelA)
    assert instance_a.param == 10

    RetrievedModelB = MODEL_REGISTRY.get('model_b')
    instance_b = RetrievedModelB(name='beta')
    assert isinstance(instance_b, ModelB)
    assert instance_b.name == 'beta'

    try:
        MODEL_REGISTRY.get('model_c')
    except KeyError as e:
        print(f"Successfully caught expected error: {e}")

    print("Registry test completed.")
