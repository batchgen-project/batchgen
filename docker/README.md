# Dockerfile Directory

This directory contains the Dockerfile and related resources for building the `batchgen` image.

## Usage

To build the Docker image, run the following command in the project root:

```bash
docker buildx build --progress=plain -f docker/Dockerfile -t batchgen:<tag> .
```

Replace `<tag>` with your desired image version.

### GPU architecture selection

The image supports both Blackwell (sm120) and Hopper (sm90) via the `GPU_ARCH` build arg
(default `blackwell`):

```bash
# Blackwell (default)
docker buildx build --progress=plain -f docker/Dockerfile -t batchgen:<tag> .

# Hopper (e.g. H20)
docker buildx build --progress=plain --build-arg GPU_ARCH=hopper -f docker/Dockerfile -t batchgen:<tag>-hopper .
```

The two builds differ only in the `batchgen_kernels` arch (sm90a vs sm120) and the FlashMLA
upstream commit. Hopper builds FlashMLA from `deepseek-ai/FlashMLA@c741387` (the
deferred-scheduling / DeepSeek-V3.2 sparse API that DeepSeek-V4-Flash requires); Blackwell
uses the older `1408756a` (it never calls FlashMLA at runtime). Everything is built from
public source — no prebuilt binaries are vendored. Both builds also include `tilelang` +
`fast_hadamard_transform` (built from GitHub source), which the V4-Flash sparse-prefill
path requires.

### Building from inside China (CN mirrors)

The default upstream sources (`download.pytorch.org`, `files.pythonhosted.org`,
recursive GitHub submodule clones) are slow or flaky from CN. The build is
parameterized so you can point it at fast mirrors without editing the Dockerfile.
Defaults are the official sources, so non-CN builds are unaffected.

```bash
docker buildx build --progress=plain --build-arg GPU_ARCH=hopper \
  --build-arg TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu129 \
  --build-arg UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  -f docker/Dockerfile -t batchgen:<tag>-hopper .
```

- `TORCH_FIND_LINKS`: flat wheel mirror for the `torch==2.9.0+cu129` install
  (uv `--find-links`). Aliyun serves the CUDA wheels at multi-MB/s vs ~127KB/s
  from `download.pytorch.org`.
- `UV_DEFAULT_INDEX`: PyPI mirror for all remaining `uv pip install` steps
  (uv ignores `PIP_INDEX_URL`); without it the `flashinfer-python` install
  crawls on `files.pythonhosted.org`.
- GitHub clones (FlashMLA/DeepGEMM + cutlass) are hardened in-Dockerfile with
  HTTP/1.1 + a retry loop to survive the intermittent HTTP2/TLS framing errors
  seen from CN; no build arg needed.

You can also directly build and push the image to a container registry by adding the `--push` flag:

```bash
docker buildx build --progress=plain -f docker/Dockerfile -t batchgen:<tag> --push .
```

### Viewing Intermediate Output

To check intermediate output during the build process, you can use `tee` to save the logs to a file:

```bash
docker buildx build --progress=plain -f docker/Dockerfile -t batchgen:<tag> . 2>&1 | tee build.log
```

This will display the output in the terminal and also write it to `build.log` for later review.

### Triggering CI Builds

To initiate a CI build for the Docker image and publish it to a container registry, use the following steps:

```bash
git tag -a <tag> -m "Release <tag>"
git push origin <tag>
```

This process creates an annotated tag and pushes it to the remote repository. Pushing the tag typically triggers the CI/CD pipeline, which will build and deploy the Docker image to the specified registry.

## Contents

- `Dockerfile`: Instructions for building the `batchgen` image.
- `v4_h20_rebuild_and_launch.sh`: One-stop runbook to rebuild the Hopper/H20
  image from current source and run a full DeepSeek-V4-Flash launch on 4x H20
  (`build` / `launch` / `wait` / `smoke` / `mmlu` / `stop`). Encodes the known
  H20 launch gotchas (512G `--shm-size`, stale-shm cleanup, sm-aware env flags).
- Other supporting files for the Docker build process.

## Notes

- Make sure Docker and Buildx are installed and configured on your system.
- Specify the image tag (`<tag>`) as needed for your versioning.