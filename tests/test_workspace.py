import json

from pygt import ExportSpec, Workspace, WorkspaceRepository
from pygt.workspace import KeepRecentPolicy, WorkspaceStatus


def test_workspace_contains_required_files(
    workspace: Workspace, export_spec1: ExportSpec, export_spec2: ExportSpec
):
    assert workspace.path.exists()
    assert workspace.model_path.exists()
    assert workspace.results_dir.exists()
    assert workspace.export_configs_dir.exists()
    assert workspace.logs_dir.exists()
    assert workspace.metadata_path.exists()
    assert workspace.export_config_path(export_spec1).exists()
    assert workspace.export_config_path(export_spec2).exists()


def test_workspace_cleanup(workspace: Workspace):
    workspace.cleanup()
    assert not workspace.path.exists()


def read_workspace_status(workspace: Workspace) -> str:
    return json.loads(workspace.metadata_path.read_text())["status"]


def test_workspace_status_after_build(workspace: Workspace):
    assert workspace.status == WorkspaceStatus.CREATED
    assert read_workspace_status(workspace) == WorkspaceStatus.CREATED


def test_workspace_status_after_marking_ready(workspace: Workspace):
    workspace.mark_ready()
    assert workspace.status == WorkspaceStatus.READY
    assert read_workspace_status(workspace) == WorkspaceStatus.READY


def test_workspace_status_after_marking_running(workspace: Workspace):
    workspace.mark_running()
    assert workspace.status == WorkspaceStatus.RUNNING
    assert read_workspace_status(workspace) == WorkspaceStatus.RUNNING


def test_workspace_status_after_marking_completed(workspace: Workspace):
    workspace.mark_completed()
    assert workspace.status == WorkspaceStatus.COMPLETED
    assert read_workspace_status(workspace) == WorkspaceStatus.COMPLETED


def test_workspace_status_after_marking_failed(workspace: Workspace):
    workspace.mark_failed()
    assert workspace.status == WorkspaceStatus.FAILED
    assert read_workspace_status(workspace) == WorkspaceStatus.FAILED


def test_workspace_status_after_marking_exported(workspace: Workspace):
    workspace.mark_exported()
    assert workspace.status == WorkspaceStatus.EXPORTED
    assert read_workspace_status(workspace) == WorkspaceStatus.EXPORTED


def test_workspace_from_metadata_file(workspace: Workspace):
    meta = json.loads(workspace.metadata_path.read_text())
    assert Workspace.from_metadata(meta) == workspace


def test_repository_refresh(repo: WorkspaceRepository):
    assert len(repo._workspaces) == 0
    repo.refresh()
    assert len(repo) == 4


def test_repository_to_sorted_list(
    repo: WorkspaceRepository, list_of_workspaces: list[Workspace]
):
    repo.refresh()
    assert repo.to_sorted_list() == list_of_workspaces


def test_repository_to_list(
    repo: WorkspaceRepository, list_of_workspaces: list[Workspace]
):
    repo.refresh()
    assert all([ws in list_of_workspaces for ws in repo.to_list()])


def test_repository_latest(
    repo: WorkspaceRepository, list_of_workspaces: list[Workspace]
):
    repo.refresh()
    assert repo.latest() == list_of_workspaces[-1]


def test_repository_oldest(
    repo: WorkspaceRepository, list_of_workspaces: list[Workspace]
):
    repo.refresh()
    assert repo.oldest() == list_of_workspaces[0]


def test_repository_clean_oldest(
    repo: WorkspaceRepository, list_of_workspaces: list[Workspace]
):
    repo.refresh()
    repo.clean_oldest()
    assert not list_of_workspaces[0].path.exists()
    repo.refresh()
    assert len(repo) == 3


def test_repository_clean_all(
    repo: WorkspaceRepository, list_of_workspaces: list[Workspace]
):
    repo.refresh()
    repo.clean_all()
    not_present = []
    for workspace in list_of_workspaces:
        not_present.append((not workspace.path.exists()))
    assert all(not_present)


def test_default_keep_all_policy(repo: WorkspaceRepository):
    repo.refresh()
    repo.apply_retention_policy()
    assert len(repo) == 4


def test_keep_recent_policy(repo: WorkspaceRepository):
    repo.refresh()
    repo.set_retention_policy(KeepRecentPolicy(keep=2))
    assert len(repo) == 4
    repo.apply_retention_policy()
    assert len(repo) == 2
    repo.apply_retention_policy()
    assert len(repo) == 2
