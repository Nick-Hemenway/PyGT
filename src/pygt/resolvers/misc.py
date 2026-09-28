from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from optwrap.context import Context
from optwrap.resolvers import Resolver


@dataclass
class ScalarExpand(Resolver):
    name: str
    count: int

    def resolve(self, context: Context) -> NDArray:
        return np.full(self.count, getattr(context, self.name))
