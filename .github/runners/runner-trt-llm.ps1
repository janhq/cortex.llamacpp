# PowerShell EntryPoint for GitHub Actions Runner

$runnerName = (hostname.exe).Trim()

# Remove existing actions-runner folder if it exists
if (Test-Path -Path C:/workspace/actions-runner) {
    Remove-Item -Recurse -Force -Path C:/workspace/actions-runner
}

# Get the latest version of the GitHub Actions runner
$response = Invoke-RestMethod -Uri https://api.github.com/repos/actions/runner/releases/latest
$env:RUNNER_VERSION = $response.tag_name.TrimStart('v')

# Download and install the runner
cmd.exe /c curl -L -o runner.zip https://github.com/actions/runner/releases/download/v$env:RUNNER_VERSION/actions-runner-win-x64-$env:RUNNER_VERSION.zip

Expand-Archive -Path C:/workspace/runner.zip -DestinationPath C:/workspace/actions-runner
Remove-Item -Path C:/workspace/runner.zip

# Navigate to the runner directory
Set-Location -Path C:/workspace/actions-runner

# Configure the runner
& .\config.cmd --unattended --replace --url https://github.com/$env:RUNNER_REPO `
    --pat $env:RUNNER_PAT `
    --runnergroup $env:RUNNER_GROUP `
    --labels $env:RUNNER_LABELS `
    --work $env:RUNNER_WORKDIR `
    --name $runnerName

# Cleanup logic
$cleanupScript = {
    Write-Host "Removing runner..."
    & .\config.cmd remove --unattended --pat $env:RUNNER_PAT
}

# Register cleanup traps
Register-EngineEvent PowerShell.Exiting -Action $cleanupScript

# Start the runner
& .\run.cmd
