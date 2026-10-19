from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from .exceptions import NonExistentWorkSpaceError
from .output_spec import ExportSpec
from .util import sanitize_path

logger = logging.getLogger(__name__)


class WorkspaceStatus(StrEnum):
    CREATED: str = "created"
    READY: str = "ready"
    RUNNING: str = "running"
    COMPLETED: str = "completed"
    FAILED: str = "failed"
    EXPORTED: str = "exported"


def make_id(dt: datetime):
    time_str = dt.strftime(r"%Y%m%d_%H%M%S")[2:]
    return f"{time_str}_{uuid4().hex[:4]}"


def _remove_readonly(func, path, excinfo):
    import stat

    # Clear the read-only bit and re-run the deletion function
    os.chmod(path, stat.S_IWRITE)
    func(path)


@dataclass
class Workspace:
    root: Path
    run_id: str
    model_name: str
    created: datetime
    status: str | None = None
    requires_export: bool = False

    @property
    def path(self) -> Path:
        return self.root / self.run_id

    @property
    def model_path(self) -> Path:
        return self.path / self.model_name

    @property
    def param_file_path(self) -> Path:
        return self.path / "params.txt"

    @property
    def param_metadata_file_path(self) -> Path:
        return self.param_file_path.with_suffix(".meta")

    @property
    def results_dir(self) -> Path:
        return self.path / "results"

    @property
    def export_configs_dir(self) -> Path:
        return self.path

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def metadata_path(self) -> Path:
        return self.path / "workspace.json"

    def export_config_path(self, spec: ExportSpec) -> Path:
        return self.export_configs_dir / spec.config_source.name

    def export_output_path(self, spec: ExportSpec) -> Path:
        return self.results_dir / spec.output_filename

    def cleanup(self):
        shutil.rmtree(self.path, onexc=_remove_readonly)

    @property
    def _metadata(self):
        return {
            "root": f"{self.root}",
            "run_id": f"{self.run_id}",
            "model_name": f"{self.model_name}",
            "created": f"{self.created.isoformat()}",
            "status": self.status,
            "requires_export": self.requires_export,
        }

    def _write_metadata(self):
        tmp = self.metadata_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._metadata, indent=2))
        tmp.replace(self.metadata_path)

    @classmethod
    def from_metadata(cls, metadata: dict) -> "Workspace":
        return cls(
            root=Path(metadata["root"]),
            run_id=metadata["run_id"],
            model_name=metadata["model_name"],
            created=datetime.fromisoformat(metadata["created"]),
            status=metadata["status"],
            requires_export=metadata["requires_export"],
        )

    def set_status(self, status: WorkspaceStatus) -> None:
        status_old = self.status
        self.status = status
        self._write_metadata()
        logger.info("Workspace status updated: %s ---> %s", status_old, self.status)

    def mark_created(self):
        self.set_status(WorkspaceStatus.CREATED)

    def mark_ready(self):
        self.set_status(WorkspaceStatus.READY)

    def mark_running(self):
        self.set_status(WorkspaceStatus.RUNNING)

    def mark_completed(self):
        self.set_status(WorkspaceStatus.COMPLETED)

    def mark_failed(self):
        self.set_status(WorkspaceStatus.FAILED)

    def mark_exported(self):
        self.set_status(WorkspaceStatus.EXPORTED)


@dataclass
class WorkspaceFactory:
    model_path: Path | str
    root: Path | str
    mode: Literal["copy", "hardlink"] = "hardlink"

    def __post_init__(self):
        self.model_path = sanitize_path(self.model_path)
        self.root = sanitize_path(self.root)

    def build(self, export_spec: ExportSpec | list[ExportSpec] | None = None):
        # Construct workspace object that defines all paths
        now = datetime.now()
        id_ = make_id(dt=now)
        ws = Workspace(
            root=self.root,
            run_id=id_,
            model_name=self.model_path.name,
            created=now,
            requires_export=(export_spec is not None),
        )

        # Actually build the paths on the file system
        to_build = ["path", "results_dir", "logs_dir", "export_configs_dir"]

        for dir_ in to_build:
            getattr(ws, dir_).mkdir(parents=True, exist_ok=True)

        # Copy all required source files to the workspace
        new_model_path = ws.path / self.model_path.name
        self._copy(src=self.model_path, dst=new_model_path)

        if export_spec is not None:
            if isinstance(export_spec, ExportSpec):
                export_spec = [export_spec]
            for spec in export_spec:
                new_config = ws.export_configs_dir / spec.config_source.name
                self._copy(src=spec.config_source, dst=new_config)

        # Mark work space as created
        # this automatically triggers writing the metadata file
        logger.info("Workspace created: %s", ws.path)
        ws.mark_created()
        return ws

    def _copy(self, src: Path, dst: Path) -> None:
        if self.mode.lower() == "hardlink":
            dst.hardlink_to(src)
        elif self.mode.lower() == "copy":
            shutil.copy2(src=src, dst=dst)
        else:
            raise ValueError("Invalid value provided for copy mode")


class RetentionPolicy(Protocol):
    def select_for_deletion(self, workspaces: list[Workspace]) -> list[Workspace]: ...


@dataclass
class KeepRecentPolicy:
    keep: int = 10

    def select_for_deletion(self, workspaces: list[Workspace]) -> list[Workspace]:
        lst = sorted(workspaces, key=lambda x: x.created)
        return lst[0 : -self.keep]


@dataclass
class KeepAllPolicy:
    def select_for_deletion(self, workspaces: list[Workspace]) -> list[Workspace]:
        return []


class WorkspaceRepository:
    def __init__(
        self, root: Path | str, retention_policy: RetentionPolicy | None = None
    ):
        self.root = sanitize_path(root)
        if retention_policy is None:
            retention_policy = KeepAllPolicy()
        self.retention_policy = retention_policy
        self._workspaces: dict[str, Workspace] = {}

    def set_retention_policy(self, retention_policy: RetentionPolicy | None = None):
        if retention_policy is None:
            retention_policy = KeepAllPolicy()
        self.retention_policy = retention_policy

    def refresh(self):
        logger.debug("WorspaceRepository being refreshed")
        self._workspaces.clear()
        logger.debug("All workspaces cleared from repository")
        for directory in self.root.iterdir():
            metadata_path = directory / "workspace.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text())
            workspace = Workspace.from_metadata(metadata)
            self._workspaces[workspace.run_id] = workspace
        logger.debug("Discovered workspaces %s", list(self._workspaces.keys()))

    def apply_retention_policy(self):
        to_delete = self.retention_policy.select_for_deletion(self.to_list())
        for workspace in to_delete:
            logger.info(
                "Removing workspace %s due to retention policy", workspace.run_id
            )
            workspace.cleanup()
        self.refresh()

    def to_sorted_list(self) -> list[Workspace]:
        return sorted(self._workspaces.values(), key=lambda x: x.created)

    def to_list(self) -> list[Workspace]:
        return list(self._workspaces.values())

    def latest(self) -> Workspace:
        try:
            return self.to_sorted_list()[-1]
        except IndexError as e:
            raise NonExistentWorkSpaceError() from e

    def previous(self) -> Workspace:
        try:
            return self.to_sorted_list()[-2]
        except IndexError as e:
            raise NonExistentWorkSpaceError() from e

    def oldest(self) -> Workspace:
        try:
            return self.to_sorted_list()[0]
        except IndexError as e:
            raise NonExistentWorkSpaceError() from e

    def clean_all(self) -> None:
        for workspace in self._workspaces.values():
            workspace.cleanup()

    def clean_oldest(self, n: int = 1):
        ws = self.to_sorted_list()
        for i in range(n):
            ws[i].cleanup()

    def __len__(self) -> int:
        return len(self._workspaces)
