from __future__ import annotations

import enum
import uuid
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, List, Sequence, TypeVar, final

import graphene
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from ai.backend.common.types import VFolderHostPermission

from .group import AssocGroupUserRow, GroupRow, UserRoleInProject
from .user import UserRole

if TYPE_CHECKING:
    from .gql import GraphQueryContext


__all__: Sequence[str] = (
    "PredefinedAtomicPermission",
    "get_all_permissions",
)


class BaseACLPermission(enum.StrEnum):
    pass


ACLPermissionType = TypeVar("ACLPermissionType", bound=BaseACLPermission)


@dataclass
class ClientContext:
    db_conn: AsyncConnection

    domain_name: str
    user_id: uuid.UUID
    user_role: UserRole

    project_ctx: Mapping[uuid.UUID, UserRoleInProject] | None = None

    async def get_or_init_project_ctx(self) -> Mapping[uuid.UUID, UserRoleInProject]:
        if self.project_ctx is None:
            if self.user_role in (UserRole.SUPERADMIN, UserRole.ADMIN):
                role_in_project = UserRoleInProject.ADMIN
            else:
                role_in_project = UserRoleInProject.USER

            if self.user_role == UserRole.SUPERADMIN:
                stmt = (
                    sa.select(AssocGroupUserRow)
                    .select_from(AssocGroupUserRow)
                    .where(AssocGroupUserRow.user_id == self.user_id)
                )
            else:
                stmt = (
                    sa.select(AssocGroupUserRow)
                    .select_from(sa.join(AssocGroupUserRow, GroupRow))
                    .where(
                        (AssocGroupUserRow.user_id == self.user_id)
                        & (GroupRow.domain_name == self.domain_name)
                    )
                )
            async with AsyncSession(self.db_conn) as db_session:
                self.project_ctx = {
                    row.group_id: role_in_project for row in await db_session.scalars(stmt)
                }
        return self.project_ctx


class BaseACLScope:
    pass


@dataclass(frozen=True)
class DomainScope(BaseACLScope):
    domain_name: str


@dataclass(frozen=True)
class ProjectScope(BaseACLScope):
    project_id: uuid.UUID


@dataclass(frozen=True)
class UserScope(BaseACLScope):
    user_id: uuid.UUID


# Extra ACL scope is used to address ACL object specific scopes
# such as registries for images, scaling groups for agents, storage hosts for vfolders etc.
ExtraACLScopeName = str
ExtraACLScopeID = Any
ExtraACLScopeType = Mapping[ExtraACLScopeName, ExtraACLScopeID]


@dataclass(frozen=True)
class ACLObjectScope:
    base_scope: BaseACLScope
    extra_scopes: ExtraACLScopeType | None = None


ACLObjectType = TypeVar("ACLObjectType")
ACLObjectIDType = TypeVar("ACLObjectIDType")


@dataclass
class AbstractACLPermissionContext(
    Generic[ACLPermissionType, ACLObjectType, ACLObjectIDType], metaclass=ABCMeta
):
    """
    Define ACL permissions under given User, Project or Domain scopes.
    Each field of this class represents a mapping of ["accessible scope id", "permissions under the scope"].
    For example, `project` field has a mapping of ["accessible project id", "permissions under the project"].
    {
        "PROJECT_A_ID": {"READ", "WRITE", "DELETE"}
        "PROJECT_B_ID": {"READ"}
    }

    `additional` and `overridden` fields have a mapping of ["ACL object id", "permissions applied to the object"].
    `additional` field is used to add permissions to specific ACL objects. It can be used for admins.
    `overridden` field is used to address exceptional cases such as permission overriding or cover other scopes(scaling groups or storage hosts etc).
    """

    user: Mapping[uuid.UUID, frozenset[ACLPermissionType]]
    project: Mapping[uuid.UUID, frozenset[ACLPermissionType]]
    domain: Mapping[str, frozenset[ACLPermissionType]]

    additional: Mapping[ACLObjectIDType, frozenset[ACLPermissionType]]
    overridden: Mapping[ACLObjectIDType, frozenset[ACLPermissionType]]

    def filter_by_permission(self, permission_to_include: ACLPermissionType) -> None:
        self.user = {
            uid: permissions
            for uid, permissions in self.user.items()
            if permission_to_include in permissions
        }
        self.project = {
            pid: permissions
            for pid, permissions in self.project.items()
            if permission_to_include in permissions
        }
        self.domain = {
            dname: permissions
            for dname, permissions in self.domain.items()
            if permission_to_include in permissions
        }
        self.additional = {
            obj_id: permissions
            for obj_id, permissions in self.additional.items()
            if permission_to_include in permissions
        }
        self.overridden = {
            obj_id: permissions
            for obj_id, permissions in self.overridden.items()
            if permission_to_include in permissions
        }

    @abstractmethod
    async def _build_query(self) -> sa.sql.Select | None:
        pass

    @final
    async def build_query(
        self, permission_to_include: ACLPermissionType | None = None
    ) -> sa.sql.Select | None:
        if permission_to_include is not None:
            self.filter_by_permission(permission_to_include)
        return await self._build_query()

    @abstractmethod
    async def determine_permission_on_obj(
        self, acl_obj: ACLObjectType
    ) -> frozenset[ACLPermissionType]:
        """
        Determine permissions applied to the given ACL object based on the fields in this class.
        """
        pass


ACLPermissionContextType = TypeVar("ACLPermissionContextType", bound=AbstractACLPermissionContext)


class AbstractACLPermissionContextBuilder(Generic[ACLPermissionContextType], metaclass=ABCMeta):
    @classmethod
    async def build(
        cls,
        db_session: AsyncSession,
        ctx: ClientContext,
        target_scope: ACLObjectScope,
    ) -> ACLPermissionContextType:
        match target_scope.base_scope:
            case UserScope(user_id=user_id):
                result = await cls._build_in_user_scope(
                    db_session, ctx, user_id, extra_target_scopes=target_scope.extra_scopes
                )
            case ProjectScope(project_id=project_id):
                result = await cls._build_in_project_scope(
                    db_session, ctx, project_id, extra_target_scopes=target_scope.extra_scopes
                )
            case DomainScope(domain_name=domain_name):
                result = await cls._build_in_domain_scope(
                    db_session, ctx, domain_name, extra_target_scopes=target_scope.extra_scopes
                )
            case _:
                raise RuntimeError(f"invalid ACL scope `{target_scope}`")
        return result

    @classmethod
    @abstractmethod
    async def _build_in_user_scope(
        cls,
        db_session: AsyncSession,
        ctx: ClientContext,
        user_id: uuid.UUID,
        *,
        extra_target_scopes: ExtraACLScopeType | None,
    ) -> ACLPermissionContextType:
        pass

    @classmethod
    @abstractmethod
    async def _build_in_project_scope(
        cls,
        db_session: AsyncSession,
        ctx: ClientContext,
        project_id: uuid.UUID,
        *,
        extra_target_scopes: ExtraACLScopeType | None,
    ) -> ACLPermissionContextType:
        pass

    @classmethod
    @abstractmethod
    async def _build_in_domain_scope(
        cls,
        db_session: AsyncSession,
        ctx: ClientContext,
        domain_name: str,
        *,
        extra_target_scopes: ExtraACLScopeType | None,
    ) -> ACLPermissionContextType:
        pass


def get_all_vfolder_host_permissions() -> List[str]:
    return [perm.value for perm in VFolderHostPermission]


def get_all_permissions() -> Mapping[str, Any]:
    return {
        "vfolder_host_permission_list": get_all_vfolder_host_permissions(),
    }


class PredefinedAtomicPermission(graphene.ObjectType):
    vfolder_host_permission_list = graphene.List(lambda: graphene.String)

    async def resolve_vfolder_host_permission_list(self, info: graphene.ResolveInfo) -> List[str]:
        return get_all_vfolder_host_permissions()

    @classmethod
    async def load_all(
        cls,
        graph_ctx: GraphQueryContext,
    ) -> PredefinedAtomicPermission:
        return cls(**get_all_permissions())
