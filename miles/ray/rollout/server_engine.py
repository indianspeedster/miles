from __future__ import annotations

import logging

import ray
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


# NOTE: currently it is almost a dataclass without encapsulation;
#       ideally, it may encapsulate all logic and ensure state transition only happens after internal actions,
#       and no external code can touch its internals
class ServerEngine:
    def __init__(self):
        self._state = _StateStopped()

    def mark_allocated(self, actor_handle: ray.actor.ActorHandle):
        self._change_state("mark_allocated", _StateStopped, _StateAllocated(actor_handle=actor_handle))

    def mark_stopped(self):
        self._change_state("mark_stopped", (_StateStopped, _StateAllocated), _StateStopped())

    @property
    def actor_handle(self) -> ray.actor.ActorHandle:
        assert isinstance(self._state, _StateAllocated)
        return self._state.actor_handle

    @property
    def is_allocated(self) -> bool:
        return isinstance(self._state, _StateAllocated)

    # TODO: unify w/ trainer `change_state`
    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[_State] | tuple[type[_State], ...],
        new_state: _State,
    ) -> None:
        logger.info(f"{debug_name} start old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self._state=}"
        self._state = new_state
        logger.info(f"{debug_name} end new={self._state}")


# ------------------------- states -----------------------------


class _StateBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class _StateStopped(_StateBase):
    pass


class _StateAllocated(_StateBase):
    actor_handle: ray.actor.ActorHandle


_State = _StateStopped | _StateAllocated
