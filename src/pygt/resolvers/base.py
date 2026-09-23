from typing import Any, Protocol, runtime_checkable

from ..simdata import CaseData


@runtime_checkable
class CaseDataResolver(Protocol):
    def resolve(self, case: CaseData) -> Any: ...
