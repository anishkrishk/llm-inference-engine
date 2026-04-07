.PHONY: help gen-proto test fmt lint clean

PYTHON ?= python
PROTO_DIR := proto
GEN_DIR := src/generated

help:
	@echo "Targets:"
	@echo "  gen-proto   Generate Python gRPC stubs from proto/inference.proto"
	@echo "  test        Run pytest"
	@echo "  fmt         Run ruff format"
	@echo "  lint        Run ruff check"
	@echo "  clean       Remove generated stubs and pycache"

gen-proto:
	@mkdir -p $(GEN_DIR)
	$(PYTHON) -m grpc_tools.protoc \
		-I $(PROTO_DIR) \
		--python_out=$(GEN_DIR) \
		--grpc_python_out=$(GEN_DIR) \
		$(PROTO_DIR)/inference.proto
	@touch $(GEN_DIR)/__init__.py
	@# protoc emits absolute imports like 'import inference_pb2' which break when
	@# the file lives inside a package. Patch them to package-relative imports.
	@sed -i 's/^import inference_pb2/from . import inference_pb2/' $(GEN_DIR)/inference_pb2_grpc.py
	@echo "Generated stubs in $(GEN_DIR)"

test:
	$(PYTHON) -m pytest

fmt:
	$(PYTHON) -m ruff format src tests benchmarks scripts

lint:
	$(PYTHON) -m ruff check src tests benchmarks scripts

clean:
	rm -rf $(GEN_DIR)/inference_pb2*.py
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
