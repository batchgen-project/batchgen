# Dockerfile Directory

This directory contains the Dockerfile and related resources for building the `moe-gen` image.

## Usage

To build the Docker image, run the following command in the project root:

```bash
docker buildx build --progress=plain -f docker/Dockerfile -t moe-gen:<tag> .
```

Replace `<tag>` with your desired image version.

### Viewing Intermediate Output

To check intermediate output during the build process, you can use `tee` to save the logs to a file:

```bash
docker buildx build --progress=plain -f docker/Dockerfile -t moe-gen:<tag> . 2>&1 | tee build.log
```

This will display the output in the terminal and also write it to `build.log` for later review.

## Contents

- `Dockerfile`: Instructions for building the `moe-gen` image.
- Other supporting files for the Docker build process.

## Notes

- Make sure Docker and Buildx are installed and configured on your system.
- Specify the image tag (`<tag>`) as needed for your versioning.