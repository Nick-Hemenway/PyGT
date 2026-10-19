from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from optwrap.context import Context
from optwrap.resolvers import AbstractResolver, Resolvable, as_resolver


@dataclass
class ScalarExpand(AbstractResolver[Context, NDArray]):
    name: Resolvable[Context, float]
    count: Resolvable[Context, int]

    def resolve(self, context: Context) -> NDArray:
        count = as_resolver(self.count).resolve(context)
        value = as_resolver(self.name).resolve(context)
        return np.full(count, value)
