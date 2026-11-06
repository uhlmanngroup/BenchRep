"""Check tutorial modules, authored configs and required assets in a built wheel."""

from hashlib import file_digest
from pathlib import Path
import sys
from zipfile import ZipFile

examples = Path(__file__).resolve().parents[2] / "examples"
files = {examples / "__init__.py"}
for pattern in ("usage/*.py", "configs/*.yaml", "configs/*/*.yaml", "assets/**/*"):
    files.update(path for path in examples.glob(pattern) if path.is_file())

standalone = examples / "assets/09_standalone_evaluation_workflow"
files.add(standalone / "anndata.h5ad")
for name in ("input.pt", "reconstruction.pt", "obs.pt", "reconstruction_export_metadata.pt"):
    files.add(standalone / "reconstruction_bundle" / name)

errors = []
with ZipFile(sys.argv[1]) as wheel:
    members = set(wheel.namelist())
    for path in sorted(files):
        member = "benchrep/examples/" + path.relative_to(examples).as_posix()
        if member not in members:
            errors.append(f"Missing from wheel: {member}")
        elif not path.is_file():
            errors.append(f"Missing from checkout: {path}")
        else:
            with wheel.open(member) as packaged, path.open("rb") as source:
                if file_digest(packaged, "sha256").digest() != file_digest(source, "sha256").digest():
                    errors.append(f"Contents differ: {member}")

if errors:
    raise SystemExit("\n".join(errors))
print(f"Verified {len(files)} example files.")
