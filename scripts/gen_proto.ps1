Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path "$PSScriptRoot/.."
$protoDir = Join-Path $repoRoot "proto"
$outDir = Join-Path $repoRoot "src/generated"

if (!(Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

python -m grpc_tools.protoc `
  -I "$protoDir" `
  --python_out="$outDir" `
  --grpc_python_out="$outDir" `
  "$protoDir/inference.proto"

$initFile = Join-Path $outDir "__init__.py"
if (!(Test-Path $initFile)) {
    New-Item -ItemType File -Path $initFile | Out-Null
}

Write-Host "Generated gRPC Python stubs in $outDir"