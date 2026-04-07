#!/usr/bin/env bash
# Regenerate Python gRPC stubs from proto/inference.proto.
# Mirrors scripts/gen_proto.ps1 for use under WSL2 / Linux / CI.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
proto_dir="$repo_root/proto"
out_dir="$repo_root/src/generated"

mkdir -p "$out_dir"

python -m grpc_tools.protoc \
    -I "$proto_dir" \
    --python_out="$out_dir" \
    --grpc_python_out="$out_dir" \
    "$proto_dir/inference.proto"

# Make the generated files importable as a package.
touch "$out_dir/__init__.py"

# protoc generates `import inference_pb2` (absolute), which fails when the
# stubs live inside a package. Rewrite to a package-relative import.
sed -i 's/^import inference_pb2/from . import inference_pb2/' \
    "$out_dir/inference_pb2_grpc.py"

echo "Generated gRPC Python stubs in $out_dir"
