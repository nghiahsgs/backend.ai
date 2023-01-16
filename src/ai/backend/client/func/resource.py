from typing import Sequence

from ..request import Request
from .base import BaseFunction, api_function

__all__ = "Resource"


class Resource(BaseFunction):
    """
    Provides interactions with resource.
    """

    @api_function
    @classmethod
    async def list(cls):
        """
        Lists all resource presets.
        """
        rqst = Request("GET", "/resource/presets")
        async with rqst.fetch() as resp:
            return await resp.json()

    @api_function
    @classmethod
    async def check_presets(cls):
        """
        Lists all resource presets in the current scaling group with additiona
        information.
        """
        rqst = Request("POST", "/resource/check-presets")
        async with rqst.fetch() as resp:
            return await resp.json()

    @api_function
    @classmethod
    async def get_docker_registries(cls):
        """
        Lists all registered docker registries.
        """
        rqst = Request("GET", "/config/docker-registries")
        async with rqst.fetch() as resp:
            return await resp.json()

    @api_function
    @classmethod
    async def usage_per_month(cls, month: str, project_ids: Sequence[str]):
        """
        Get usage statistics for projects specified by `project_ids` at specific `month`.

        :param month: The month you want to get the statistics (yyyymm).
        :param project_ids: Groups IDs to be included in the result.
        """
        rqst = Request("GET", "/resource/usage/month")
        rqst.set_json(
            {
                "month": month,
                "project_ids": project_ids,
            }
        )
        async with rqst.fetch() as resp:
            return await resp.json()

    @api_function
    @classmethod
    async def usage_per_period(cls, project_id: str, start_date: str, end_date: str):
        """
        Get usage statistics for a project specified by `project_id` for time betweeen
        `start_date` and `end_date`.

        :param start_date: start date in string format (yyyymmdd).
        :param end_date: end date in string format (yyyymmdd).
        :param project_id: Groups ID to list usage statistics.
        """
        rqst = Request("GET", "/resource/usage/period")
        rqst.set_json(
            {
                "project_id": project_id,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        async with rqst.fetch() as resp:
            return await resp.json()

    @api_function
    @classmethod
    async def get_resource_slots(cls):
        """
        Get supported resource slots of Backend.AI server.
        """
        rqst = Request("GET", "/config/resource-slots")
        async with rqst.fetch() as resp:
            return await resp.json()

    @api_function
    @classmethod
    async def get_vfolder_types(cls):
        rqst = Request("GET", "/config/vfolder-types")
        async with rqst.fetch() as resp:
            return await resp.json()

    @api_function
    @classmethod
    async def recalculate_usage(cls):
        rqst = Request("POST", "/resource/recalculate-usage")
        async with rqst.fetch() as resp:
            return await resp.json()

    @api_function
    @classmethod
    async def user_monthly_stats(cls):
        rqst = Request("GET", "/resource/stats/user/month")
        async with rqst.fetch() as resp:
            return await resp.json()

    @api_function
    @classmethod
    async def admin_monthly_stats(cls):
        rqst = Request("GET", "/resource/stats/admin/month")
        async with rqst.fetch() as resp:
            return await resp.json()
