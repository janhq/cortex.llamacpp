#!/bin/bash

RUNNER_REPO=$RUNNER_REPO
RUNNER_PAT=$RUNNER_PAT
RUNNER_GROUP=$RUNNER_GROUP
RUNNER_LABELS=$RUNNER_LABELS
RUNNER_NAME=$(hostname)

# Get the latest version of the GitHub Actions runner
LATEST_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep -oP '(?<=\"tag_name\": \"v)[^\"]*')
OS_ARCH=$(uname -m)

if [[ "$OS_ARCH" == "x86_64" ]]; then
    ARCH="x64"
elif [[ "$OS_ARCH" == "aarch64" ]]; then
    ARCH="arm64"
else
    echo "Unsupported architecture: $OS_ARCH"
    exit 1
fi

# Download and install the GitHub Actions runner
cd /home/runner && rm -rf actions-runner && mkdir -p actions-runner && cd actions-runner
curl -O -L https://github.com/actions/runner/releases/download/v${LATEST_VERSION}/actions-runner-linux-${ARCH}-${LATEST_VERSION}.tar.gz

tar xzf ./actions-runner-linux-${ARCH}-${LATEST_VERSION}.tar.gz

# Navigate to actions-runner directory
cd /home/runner/actions-runner

./config.sh --unattended --replace --url https://github.com/${RUNNER_REPO} \
    --pat ${RUNNER_PAT} \
    --name ${RUNNER_NAME} \
    --runnergroup ${RUNNER_GROUP} \
    --labels ${RUNNER_LABELS} \
    --work /home/runner/actions-runner/_work

cleanup() {
    echo "Removing runner..."
    ./config.sh remove --unattended --pat ${RUNNER_PAT}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

./run.sh & wait $!
