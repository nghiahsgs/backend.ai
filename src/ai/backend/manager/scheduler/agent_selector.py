from __future__ import annotations

import sys
from decimal import Decimal
from typing import Optional, Sequence, override

import trafaret as t

from ai.backend.common.types import (
    AgentId,
    ResourceSlot,
    RoundRobinState,
)

from ..models import AgentRow, KernelRow, SessionRow
from .types import (
    AbstractAgentSelector,
    AbstractStateInjector,
    EtcdRoundRobinStateInjector,
    InmemoryRoundRobinStateInjector,
)
from .utils import (
    get_requested_architecture,
    sort_requested_slots_by_priority,
)


def get_num_extras(agent: AgentRow, requested_slots: ResourceSlot) -> int:
    """
    Get the number of resource slots that:
    1) are requested but zero (unused),
    2) are available in the given agent.

    This is to prefer (or not) agents with additional unused slots,
    depending on the selection strategy.
    """
    unused_slot_keys = set()
    for k, v in requested_slots.items():
        if v == Decimal(0):
            unused_slot_keys.add(k)
    num_extras = 0
    for k, v in agent.available_slots.items():
        if k in unused_slot_keys and v > Decimal(0):
            num_extras += 1

    return num_extras


class BaseAgentSelector(AbstractAgentSelector):
    @property
    @override
    def config_iv(self) -> t.Dict:
        return t.Dict({}).allow_extra("*")

    def filter_agents(
        self,
        compatible_agents: Sequence[AgentRow],
        pending_session_or_kernel: SessionRow | KernelRow,
    ) -> Sequence[AgentRow]:
        """
        Filter the agents by checking if it can host the picked session.
        """
        return [
            agent
            for agent in compatible_agents
            if (
                agent.available_slots - agent.occupied_slots
                >= pending_session_or_kernel.requested_slots
            )
        ]


class LegacyAgentSelector(BaseAgentSelector):
    @override
    async def select_agent(
        self,
        agents: Sequence[AgentRow],
        pending_session_or_kernel: SessionRow | KernelRow,
    ) -> Optional[AgentId]:
        agents = self.filter_agents(agents, pending_session_or_kernel)
        if not agents:
            return None
        requested_slots = pending_session_or_kernel.requested_slots
        resource_priorities = sort_requested_slots_by_priority(
            requested_slots, self.agent_selection_resource_priority
        )
        chosen_agent = max(
            agents,
            key=lambda agent: [
                -get_num_extras(agent, requested_slots),
                *[agent.available_slots.get(key, -sys.maxsize) for key in resource_priorities],
            ],
        )
        return chosen_agent.id


class RoundRobinAgentSelector(BaseAgentSelector):
    @override
    async def assign_agent_for_kernel(
        self,
        agents: Sequence[AgentRow],
        pending_kernel: KernelRow,
    ) -> Optional[AgentId]:
        # Note that ROUNDROBIN is not working with the multi-node multi-container session.
        # It assumes the pending session type is single-node session.
        # Otherwise, fall back to the implementation of DISPERSED.
        alternative_impl = DispersedAgentSelector(
            self.sgroup_opts,
            {},  # use the default config
            self.agent_selection_resource_priority,
            self.shared_config,
        )
        return await alternative_impl.select_agent(agents, pending_kernel)

    @override
    async def select_agent(
        self,
        agents: Sequence[AgentRow],
        pending_session_or_kernel: SessionRow | KernelRow,
    ) -> Optional[AgentId]:
        assert isinstance(pending_session_or_kernel, SessionRow)
        sgroup_name = pending_session_or_kernel.scaling_group_name
        requested_architecture = get_requested_architecture(pending_session_or_kernel)

        rr_state_injector: AbstractStateInjector
        match self.config.get("injector-type", None):
            case "etcd":
                rr_state_injector = EtcdRoundRobinStateInjector(self.shared_config)
            case "inmemory":
                rr_state_injector = InmemoryRoundRobinStateInjector()
            case _ as unknown:
                raise ValueError(f"Unknown state injector type: {unknown}")

        rr_state: RoundRobinState | None = await rr_state_injector.get_state((
            sgroup_name,
            requested_architecture,
        ))

        if rr_state is None:
            agent_start_idx = 0
        else:
            agent_start_idx = rr_state.next_index % len(agents)

        chosen_agent = None
        agents = sorted(agents, key=lambda agent: agent.id)

        for i in range(len(agents)):
            idx = (agent_start_idx + i) % len(agents)

            if (
                agents[idx].available_slots - agents[idx].occupied_slots
                >= pending_session_or_kernel.requested_slots
            ):
                chosen_agent = agents[idx]
                rr_state = RoundRobinState(next_index=(idx + 1) % len(agents))
                await rr_state_injector.put_state((sgroup_name, requested_architecture), rr_state)
                break

        if not chosen_agent:
            return None

        return chosen_agent.id


class ConcentratedAgentSelector(BaseAgentSelector):
    @override
    async def select_agent(
        self,
        agents: Sequence[AgentRow],
        pending_session_or_kernel: SessionRow | KernelRow,
    ) -> Optional[AgentId]:
        agents = self.filter_agents(agents, pending_session_or_kernel)
        if not agents:
            return None
        requested_slots = pending_session_or_kernel.requested_slots
        resource_priorities = sort_requested_slots_by_priority(
            requested_slots, self.agent_selection_resource_priority
        )
        chosen_agent = min(
            agents,
            key=lambda agent: [
                get_num_extras(agent, requested_slots),
                *[
                    (agent.available_slots - agent.occupied_slots).get(key, sys.maxsize)
                    for key in resource_priorities
                ],
            ],
        )
        return chosen_agent.id


class DispersedAgentSelector(BaseAgentSelector):
    @override
    async def select_agent(
        self,
        agents: Sequence[AgentRow],
        pending_session_or_kernel: SessionRow | KernelRow,
    ) -> Optional[AgentId]:
        agents = self.filter_agents(agents, pending_session_or_kernel)
        if not agents:
            return None
        requested_slots = pending_session_or_kernel.requested_slots
        resource_priorities = sort_requested_slots_by_priority(
            requested_slots, self.agent_selection_resource_priority
        )
        chosen_agent = max(
            agents,
            key=lambda agent: [
                -get_num_extras(agent, requested_slots),
                *[
                    (agent.available_slots - agent.occupied_slots).get(key, -sys.maxsize)
                    for key in resource_priorities
                ],
            ],
        )
        return chosen_agent.id
