from dataclasses import dataclass
from ortools.sat.python import cp_model
import random
from typing import Callable, Generic, Hashable, Iterable, TypeVar

T = TypeVar("T", bound=Hashable)


@dataclass
class Expr(Generic[T]):
    apply: Callable[[cp_model.CpModel], cp_model.IntVar]
    domain: set[T]

    def __post_init__(self) -> None:
        cache: dict[int, cp_model.IntVar] = {}

        def cached_apply(model: cp_model.CpModel) -> cp_model.IntVar:
            key = id(model)
            if key not in cache:
                cache[key] = self.apply(model)
            return cache[key]

        self.apply = cached_apply
        self.indices = {value: i for i, value in enumerate(self.domain)}

    def __eq__(self, other: "Expr[T]" | T) -> "Expr[bool]":
        def apply(model: cp_model.CpModel) -> cp_model.IntVar:
            var = self.apply(model)
            value = (
                other.apply(model)
                if isinstance(other, Expr)
                else self.indices.get(other, -1)
            )
            eq_var = model.new_bool_var(
                f"{var.name} == {value.name if isinstance(value, cp_model.IntVar) else other}"
            )
            model.add(var == value).only_enforce_if(eq_var)
            model.add(var != value).only_enforce_if(~eq_var)
            return eq_var

        return Expr(apply, {True, False})

    def __ne__(self, other: "Expr[T]" | T) -> "Expr[bool]":
        def apply(model: cp_model.CpModel) -> cp_model.IntVar:
            var = self.apply(model)
            value = (
                other.apply(model)
                if isinstance(other, Expr)
                else self.indices.get(other, -1)
            )
            ne_var = model.new_bool_var(
                f"{var.name} != {value.name if isinstance(value, cp_model.IntVar) else other}"
            )
            model.add(var != value).only_enforce_if(ne_var)
            model.add(var == value).only_enforce_if(~ne_var)
            return ne_var

        return Expr(apply, {True, False})

    def __lt__(self, other: "Expr[T]" | T) -> "Expr[bool]":
        def apply(model: cp_model.CpModel) -> cp_model.IntVar:
            var = self.apply(model)
            value = (
                other.apply(model)
                if isinstance(other, Expr)
                else self.indices.get(other, -1)
            )
            lt_var = model.new_bool_var(
                f"{var.name} < {value.name if isinstance(value, cp_model.IntVar) else other}"
            )
            model.add(var < value).only_enforce_if(lt_var)
            model.add(var >= value).only_enforce_if(~lt_var)
            return lt_var

        return Expr(apply, {True, False})

    def __le__(self, other: "Expr[T]" | T) -> "Expr[bool]":
        def apply(model: cp_model.CpModel) -> cp_model.IntVar:
            var = self.apply(model)
            value = (
                other.apply(model)
                if isinstance(other, Expr)
                else self.indices.get(other, -1)
            )
            le_var = model.new_bool_var(
                f"{var.name} <= {value.name if isinstance(value, cp_model.IntVar) else other}"
            )
            model.add(var <= value).only_enforce_if(le_var)
            model.add(var > value).only_enforce_if(~le_var)
            return le_var

        return Expr(apply, {True, False})

    def __gt__(self, other: "Expr[T]" | T) -> "Expr[bool]":
        def apply(model: cp_model.CpModel) -> cp_model.IntVar:
            var = self.apply(model)
            value = (
                other.apply(model)
                if isinstance(other, Expr)
                else self.indices.get(other, -1)
            )
            gt_var = model.new_bool_var(
                f"{var.name} > {value.name if isinstance(value, cp_model.IntVar) else other}"
            )
            model.add(var > value).only_enforce_if(gt_var)
            model.add(var <= value).only_enforce_if(~gt_var)
            return gt_var

        return Expr(apply, {True, False})

    def __ge__(self, other: "Expr[T]" | T) -> "Expr[bool]":
        def apply(model: cp_model.CpModel) -> cp_model.IntVar:
            var = self.apply(model)
            value = (
                other.apply(model)
                if isinstance(other, Expr)
                else self.indices.get(other, -1)
            )
            ge_var = model.new_bool_var(
                f"{var.name} >= {value.name if isinstance(value, cp_model.IntVar) else other}"
            )
            model.add(var >= value).only_enforce_if(ge_var)
            model.add(var < value).only_enforce_if(~ge_var)
            return ge_var

        return Expr(apply, {True, False})

    def __add__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __radd__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __sub__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __rsub__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __mul__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __rmul__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __truediv__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __rtruediv__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __floordiv__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __rfloordiv__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __mod__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __rmod__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __pow__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    def __rpow__(self, other: "Expr[T]" | T) -> "Expr[T]": ...

    # def __and__(self, other: "Expr[bool] | bool") -> "Expr[bool]": ...

    # def __rand__(self, other: "Expr[bool] | bool") -> "Expr[bool]": ...

    # def __or__(self, other: "Expr[bool] | bool") -> "Expr[bool]": ...

    # def __ror__(self, other: "Expr[bool] | bool") -> "Expr[bool]": ...

    # def __xor__(self, other: "Expr[bool] | bool") -> "Expr[bool]": ...

    # def __rxor__(self, other: "Expr[bool] | bool") -> "Expr[bool]": ...

    # def __invert__(self) -> "Expr[bool]": ...

    # def __neg__(self) -> "Expr[T]": ...

    # def __pos__(self) -> "Expr[T]": ...

    # def __abs__(self) -> "Expr[T]": ...


def test(expr1: Expr[int], expr2: Expr[int]) -> None:
    _ = expr1 or expr2


class Var(Generic[T]):
    def __init__(self, domain: Iterable[T], name: str | None = None) -> None:
        self.domain = set(domain)
        self.name = name

    def apply(self, model: cp_model.CpModel) -> cp_model.IntVar:
        return model.new_int_var(
            0,
            len(self.domain) - 1,
            name=self.name or f"var_{random.randint(0, 1000000)}",
        )

    def __repr__(self) -> str:
        return f"Var({self.domain}, {self.name})"

    def __str__(self) -> str:
        return f"Var({self.domain}, {self.name})"

    def __eq__(self, other: "Var[T]" | T) -> "Var": ...

    def __req__(self, other: "Var[T]" | T) -> "Var": ...
