from pathlib import Path

import pytest

from pygt import ExportSpec, Workspace, WorkspaceFactory, WorkspaceRepository


@pytest.fixture
def two_temp_paths(tmp_path_factory: pytest.TempdirFactory):
    p1 = tmp_path_factory.mktemp("first")
    p2 = tmp_path_factory.mktemp("second")
    return p1, p2


@pytest.fixture
def factory(two_temp_paths: tuple[Path, Path]) -> WorkspaceFactory:
    p1, p2 = two_temp_paths
    model = p1 / "madeup_model.gtm"
    model.write_text("some text")
    return WorkspaceFactory(model_path=model, root=p2)


@pytest.fixture
def export_spec1(tmp_path: Path) -> ExportSpec:
    f = tmp_path / "export_config1.cfg"
    f.write_text("export settings")
    return ExportSpec(
        name="exported",
        config_source=f,
        output_filename="export_results1.txt",
        index_name="angle",
        name_map={"col1": "temperature"},
    )


@pytest.fixture
def export_spec2(tmp_path: Path) -> ExportSpec:
    f = tmp_path / "export_config2.cfg"
    f.write_text("export settings")
    return ExportSpec(
        name="exported",
        config_source=f,
        output_filename="export_results2.txt",
        index_name="angle",
        name_map={"col1": "speed"},
    )


@pytest.fixture
def workspace(
    factory: WorkspaceFactory, export_spec1: ExportSpec, export_spec2: ExportSpec
) -> Workspace:
    return factory.build(export_spec=[export_spec1, export_spec2])


@pytest.fixture
def list_of_workspaces(factory: WorkspaceFactory) -> list[Workspace]:
    ws = []
    for _ in range(4):
        ws.append(factory.build())
    return ws


@pytest.fixture
def repo(list_of_workspaces: list[Workspace]) -> WorkspaceRepository:
    return WorkspaceRepository(root=list_of_workspaces[0].path.parent)
