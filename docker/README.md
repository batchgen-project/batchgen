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
- Other supporting files for the Docker build process.

## Notes

- Make sure Docker and Buildx are installed and configured on your system.
- Specify the image tag (`<tag>`) as needed for your versioning.