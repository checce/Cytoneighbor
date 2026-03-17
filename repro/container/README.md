# Container Reproducibility Notes

This repository provides two reproducibility recipes:

- `repro/container/Dockerfile.smn`
- `repro/container/Apptainer.smn.def`

Both recipes reconstruct the `smn` environment from:

- `repro/locks/environment-smn.yml`
- `repro/locks/pip-freeze-smn.txt`

## Build Commands

Docker:

```bash
docker build -f repro/container/Dockerfile.smn -t cytoneighbor-smn:lock .
```

Apptainer:

```bash
apptainer build cytoneighbor-smn.sif repro/container/Apptainer.smn.def
```

## Image Digest Pinning

After building, pin the final image digest with:

```bash
docker image inspect cytoneighbor-smn:lock --format='{{index .RepoDigests 0}}'
```

In environments without Docker/Apptainer, `sha256` hashes of the container recipe
files are recorded in `repro/repro_manifest.csv` and `repro/repro_manifest.json`.
