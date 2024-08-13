from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
)

import attrs
import trafaret as t

from ai.backend.common.logging import BraceStyleAdapter
from ai.backend.common.types import (
    AgentId,
    AgentSelectionStrategy,
    KernelId,
    ResourceSlot,
    SessionId,
    SlotName,
    SlotTypes,
)

from ..models import AgentRow, KernelRow, SessionRow
from ..models.scaling_group import ScalingGroupOpts
from ..registry import AgentRegistry

log = BraceStyleAdapter(logging.getLogger("ai.backend.manager.scheduler"))


def merge_resource(
    target: MutableMapping[str, Any],
    update: MutableMapping[str, Any],
) -> None:
    for k in update.keys():
        if k in target.keys():
            target[k] += update[k]
        else:
            target[k] = update[k]


@attrs.define(auto_attribs=True, slots=True)
class AgentAllocationContext:
    agent_id: Optional[AgentId]
    agent_addr: str
    scaling_group: str


@attrs.define(auto_attribs=True, slots=True)
class ScheduleDecision:
    agent_id: AgentId
    kernel_id: KernelId


@attrs.define(auto_attribs=True, slots=True)
class SchedulingContext:
    """
    Context for each scheduling decision.
    """

    registry: AgentRegistry
    known_slot_types: Mapping[SlotName, SlotTypes]


@attrs.define(auto_attribs=True, slots=True)
class KernelAgentBinding:
    kernel: KernelRow
    agent_alloc_ctx: AgentAllocationContext
    allocated_host_ports: Set[int]


@attrs.define(auto_attribs=True, slots=True)
class PredicateResult:
    passed: bool
    message: Optional[str] = None


class AbstractScheduler(metaclass=ABCMeta):
    """
    Interface for scheduling algorithms where the
    ``schedule()`` method is a pure function.
    """

    sgroup_opts: ScalingGroupOpts  # sgroup-specific config
    config: Mapping[str, Any]  # scheduler-specific config
    config_iv: t.Dict

    def __init__(self, sgroup_opts: ScalingGroupOpts, config: Mapping[str, Any]) -> None:
        self.sgroup_opts = sgroup_opts
        self.config = self.config_iv.check(config)

    @abstractmethod
    def pick_session(
        self,
        total_capacity: ResourceSlot,
        pending_sessions: Sequence[SessionRow],
        existing_sessions: Sequence[SessionRow],
    ) -> Optional[SessionId]:
        """
        Pick a session to try schedule.
        This is where the queueing semantics is implemented such as prioritization.
        """
        return None

    @abstractmethod
    def assign_agent_for_session(
        self,
        possible_agents: Sequence[AgentRow],
        pending_session: SessionRow,
        agent_selection_strategy: AgentSelectionStrategy,
        agent_selection_resource_priority: list[str],
    ) -> Optional[AgentId]:
        """
        Assign an agent for the entire session, only considering the total requested
        slots of the session.  This is used for both single-container sessions and
        single-node multi-container sessions.

        In single-node multi-container sessions, all sub-containers are spawned by
        slicing the assigned agent's resource.
        """
        return None

    @abstractmethod
    def assign_agent_for_kernel(
        self,
        possible_agents: Sequence[AgentRow],
        pending_kernel: KernelRow,
        agent_selection_strategy: AgentSelectionStrategy,
        agent_selection_resource_priority: list[str],
    ) -> Optional[AgentId]:
        """
        Assign an agent for a kernel of the session.
        This may be called multiple times for multi-node multi-container sessions.
        """
        return None
