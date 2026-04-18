# data-squirrel (data_nut_squirrel) — User Guide

## What It Is

data-squirrel is a Python framework that generates strongly-typed, auto-persisting data classes from YAML schema definitions. You define your data structure in a YAML file, run the code generator, and get a Python module with typed properties that automatically save to disk on every write.

No ORM. No SQL. No manual serialization. You write `strand.primary_structure.strand = "AUGC"` and the value persists to a YAML file immediately.

## How It Works

1. **Define** your data structure in a YAML schema file
2. **Generate** Python API classes with `make_data_squirrel_api`
3. **Use** the generated classes — every attribute write auto-persists

## The YAML Schema Format

A data-squirrel schema has two top-level sections:

### NUT Section

The `NUT` section defines the root object — the top-level container:

```yaml
NUT:
  db_info: "my_database"
  nut_main_struct:
    name: MyDataStore
    object_list:
      - name: weather
        object_type: CONTAINER
        object_info: WeatherContainer
      - name: location
        object_type: CONTAINER
        object_info: LocationContainer
```

Fields:
- `db_info` — the database/storage identifier
- `nut_main_struct.name` — the name of the root Python class
- `object_list` — list of top-level attributes, each with:
  - `name` — the attribute name
  - `object_type` — one of: `VALUE`, `CONTAINER`, `CLASS`, `list`, `dict`
  - `object_info` — for VALUE: the Python type (`int`, `float`, `str`, `bool`). For CONTAINER: the name of a definition from the DEFINITIONS section.

### DEFINITIONS Section

The `DEFINITIONS` section defines all nested container structures:

```yaml
DEFINITIONS:
  nut_containers_definitions:
    - name: WeatherContainer
      object_list:
        - name: temperature
          object_type: VALUE
          object_info: float
        - name: humidity
          object_type: VALUE
          object_info: float
        - name: wind_speed
          object_type: VALUE
          object_info: float
        - name: conditions
          object_type: VALUE
          object_info: str

    - name: LocationContainer
      object_list:
        - name: city
          object_type: VALUE
          object_info: str
        - name: latitude
          object_type: VALUE
          object_info: float
        - name: longitude
          object_type: VALUE
          object_info: float
```

Each container definition has:
- `name` — the container class name (referenced by `object_info` in the NUT section)
- `object_list` — list of attributes in this container

### Object Types

| object_type | object_info | Description |
|-------------|-------------|-------------|
| `VALUE` | `int`, `float`, `str`, `bool` | A single typed value |
| `CONTAINER` | Name of a definition | A nested structure (references a DEFINITIONS entry) |
| `CLASS` | `[ClassName, CLASS]` | A reference to another class |
| `list` | `[ItemType, CLASS]` | A list of typed items |
| `dict` | `[ValueType]` | A dictionary with string keys |

## Complete Example Schema

Here is a complete data-squirrel YAML schema for an RNA analysis data store:

```yaml
NUT:
  db_info: "rna_analysis"
  nut_main_struct:
    name: RNAStrand
    object_list:
      - name: primary_structure
        object_type: CONTAINER
        object_info: PrimaryStructure
      - name: ensemble
        object_type: CONTAINER
        object_info: Ensemble

DEFINITIONS:
  nut_containers_definitions:
    - name: PrimaryStructure
      object_list:
        - name: strand
          object_type: VALUE
          object_info: str
        - name: length
          object_type: VALUE
          object_info: int

    - name: Ensemble
      object_list:
        - name: max_energy
          object_type: CONTAINER
          object_info: Energy
        - name: mfe_structure
          object_type: VALUE
          object_info: str

    - name: Energy
      object_list:
        - name: kcal
          object_type: VALUE
          object_info: float
```

## What the Generated Python Code Looks Like

After running the code generator on the schema above, you get a Python module with classes like:

```python
from pathlib import Path

class RNAStrand(Nut):
    def __init__(self, working_folder: Path, var_name: str):
        super().__init__(
            enum_list=Nut_Attributes,
            use_db=True,
            db=None,
            var_name=var_name,
            working_folder=working_folder
        )

class Energy(CustomAttribute):
    @property
    def kcal(self) -> float:
        return self.parent.kcal_db

    @kcal.setter
    def kcal(self, value: float):
        if not isinstance(value, float):
            raise ValueError("Invalid value assignment")
        self.parent.kcal_db = value
```

## Using the Generated Classes

```python
from pathlib import Path

# Create an instance — this creates the storage directory
strand = RNAStrand(
    working_folder=Path("/path/to/data"),
    var_name="my_rna_strand"
)

# Write values — automatically persists to YAML
strand.primary_structure.strand = "AUGCAUGC"
strand.primary_structure.length = 8
strand.ensemble.max_energy.kcal = -4.2
strand.ensemble.mfe_structure = "(((...)))"

# Read values — loaded from YAML
print(strand.primary_structure.strand)    # "AUGCAUGC"
print(strand.ensemble.max_energy.kcal)    # -4.2
```

Every write goes to a YAML file in the `working_folder`. Every read loads from the same file. No explicit save/load calls needed.

## Key Classes in the Framework

| Class | File | Purpose |
|-------|------|---------|
| `Nut` | `dynamic_data_nut.py` | Base class for root objects. Manages persistence. |
| `CustomAttribute` | `dynamic_data_nut.py` | Wraps nested container structures. Intercepts reads/writes. |
| `ValuePacket` | `dynamic_data_nut.py` | Wraps individual typed values. |
| `NutObject` | `nut_yaml_objects.py` | Schema definition for a single attribute. |
| `NutContainer` | `nut_yaml_objects.py` | Schema definition for a container. |
| `NutStructure` | `nut_yaml_objects.py` | Full schema definition (NUT + containers). |
| `PythonBuild` | `nut_python_build.py` | Code generator — YAML schema to Python. |
| `YamlDataOperations` | `nut_data_manager.py` | File I/O — reads/writes YAML data files. |

## Generating the API

```bash
pip install data-nut-squirrel
make_data_squirrel_api --schema my_schema.yaml --output my_module.py
```

Or programmatically:

```python
from data_squirrel.make_single_api_file import build_shared_python_nut

build_shared_python_nut(
    yaml_path="my_schema.yaml",
    output_path="my_module.py"
)
```

## Design Principles

1. **No SQL** — data is stored as YAML files, human-readable
2. **Type-safe** — generated properties validate types at runtime
3. **Auto-persist** — every write saves immediately, no explicit flush
4. **Zero boilerplate** — define the schema, generate the code, use it
5. **Hierarchical** — nested containers mirror your data's natural structure
