"""Generic named component registry for KV-related coordinators."""

from __future__ import annotations

from typing import Any, Dict, Iterator


class ComponentCoordinator:
    """Small registry that does not assume a component protocol."""

    component_label = "component"

    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}

    def register_component(
        self,
        component_name: str,
        component: Any,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise ValueError(
                f"{type(self).__name__} does not accept component metadata"
            )
        if not component_name:
            raise ValueError(f"{self.component_label} name must be non-empty")
        if component is None:
            raise ValueError(
                f"{self.component_label} {component_name!r}: component must be set"
            )
        if component_name in self._components:
            raise ValueError(
                f"{self.component_label} already registered: {component_name}"
            )
        self._components[component_name] = component
        setattr(self, component_name, component)
        return component

    @property
    def component_names(self) -> list[str]:
        return list(self._components.keys())

    def components(self) -> Iterator[tuple[str, Any]]:
        return iter(self._components.items())

    def get_component(self, name: str) -> Any:
        try:
            return self._components[name]
        except KeyError as exc:
            raise KeyError(f"Unknown {self.component_label}: {name}") from exc

    def call_component(
        self, component_name: str, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        component = self.get_component(component_name)
        method = getattr(component, method_name)
        return method(*args, **kwargs)

    def call_all(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component_name, component in self.components():
            method = getattr(component, method_name)
            results[component_name] = method(*args, **kwargs)
        return results
