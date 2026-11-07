"""Project settings: Pydantic Settings over TOML files and environment variables.

    >>> from modephere.config import get_settings
    >>> get_settings().paths.library
    PosixPath('/.../data/library')

Priority, highest first: constructor arguments, ``MODEPHERE_*`` environment
variables (nested with ``__``, e.g. ``MODEPHERE_PATHS__LIBRARY=/scratch/lib``),
``config/data/{paths,settings}.toml``, field defaults. Relative paths are
resolved against the project root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_DATA_DIR: Path = Path(__file__).parent / "data"


def _project_root() -> Path:
    # src/modephere/config/settings.py -> project root (editable install), else
    # the working directory (installed wheel).
    candidate = Path(__file__).resolve().parents[3]
    return candidate if (candidate / "pyproject.toml").exists() else Path.cwd()


PROJECT_ROOT: Path = _project_root()


class Paths(BaseModel):
    """Filesystem layout; relative entries are resolved against the project root."""

    project_root: Path = PROJECT_ROOT
    data: Path = Path("data")
    logs: Path = Path("logs")
    library: Path = Path("data/library")
    cache: Path = Path("data/cache")
    chembl: Path = Path("data/chembl")
    pgx: Path = Path("data/pgx")

    @field_validator("data", "logs", "library", "cache", "chembl", "pgx")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else PROJECT_ROOT / v


class PubChemSettings(BaseModel):
    """PubChem PUG-REST client etiquette."""

    base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    max_requests_per_second: PositiveFloat = Field(
        default=5.0, description="PubChem's published ceiling."
    )
    timeout_s: PositiveFloat = 30.0
    batch_size: PositiveInt = Field(
        default=100, description="CIDs per property request."
    )
    user_agent: str = "modephere/0.1 (+https://pubchem.ncbi.nlm.nih.gov)"


class ChemblSettings(BaseModel):
    """The local ChEMBL SQLite release and how bioactivities are aggregated."""

    sqlite: Path | None = Field(
        default=None,
        description="chembl_<N>.db from the chembl_<N>_sqlite download.",
    )
    min_confidence: int = Field(
        default=7,
        ge=0,
        le=9,
        description=(
            "Lowest assay target confidence score aggregated into compound-target "
            "edges (7 = direct protein complex, 8 = homologous protein, "
            "9 = direct single protein). Raw activities keep every score."
        ),
    )

    @field_validator("sqlite")
    @classmethod
    def _expand(cls, v: Path | None) -> Path | None:
        return None if v is None else v.expanduser()


class KGSettings(BaseModel):
    """Neo4j connection of the knowledge graph."""

    uri: str = "bolt://localhost:7688"
    user: str = "neo4j"
    password: SecretStr = SecretStr("modephere-dev")
    database: str = "neo4j"
    batch_size: PositiveInt = 1000


class NCBISettings(BaseModel):
    """NCBI E-utilities / Variation Services client identity (``modephere.pgx``).

    ``MODEPHERE_NCBI__API_KEY`` raises the E-utilities limit from 3 to 10
    requests/s; ``MODEPHERE_NCBI__EMAIL`` is sent only when set.
    """

    api_key: SecretStr | None = None
    email: str | None = None
    tool: str = "modephere"
    timeout_s: PositiveFloat = 30.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODEPHERE_",
        env_nested_delimiter="__",
        extra="ignore",
        toml_file=[CONFIG_DATA_DIR / "paths.toml", CONFIG_DATA_DIR / "settings.toml"],
    )

    project_name: str = "Modephere"
    seed: int = 42
    log_level: str = "WARNING"
    paths: Paths = Field(default_factory=Paths)
    pubchem: PubChemSettings = Field(default_factory=PubChemSettings)
    chembl: ChemblSettings = Field(default_factory=ChemblSettings)
    kg: KGSettings = Field(default_factory=KGSettings)
    ncbi: NCBISettings = Field(default_factory=NCBISettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the settings (``get_settings.cache_clear()`` to reload)."""
    return Settings()


__all__ = [
    "PROJECT_ROOT",
    "ChemblSettings",
    "KGSettings",
    "NCBISettings",
    "Paths",
    "PubChemSettings",
    "Settings",
    "get_settings",
]
