#!/usr/bin/env python3
"""
Compile inference engine protobuf definitions into Python code.

This script uses grpcio-tools to generate *_pb2.py, *_pb2_grpc.py, and
*_pb2.pyi (type stubs) files from the inference_engine.proto definition.

Usage:
    python inference_grpc/compile_protos.py

Requirements:
    pip install grpcio-tools
"""

import sys
from pathlib import Path


def compile_protos():
    """Compile protobuf definitions."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    proto_file = script_dir / "inference_engine.proto"

    if not proto_file.exists():
        print(f"Error: Proto file not found at {proto_file}")
        return 1

    print(f"Compiling protobuf: {proto_file}")
    print(f"Output directory: {script_dir}")

    try:
        from grpc_tools import protoc

        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"--proto_path={workspace_root}",
                f"--python_out={workspace_root}",
                f"--grpc_python_out={workspace_root}",
                f"--pyi_out={workspace_root}",
                str(script_dir / "inference_engine.proto"),
            ]
        )

        if result == 0:
            for generated_file in [
                script_dir / "inference_engine_pb2.py",
                script_dir / "inference_engine_pb2_grpc.py",
                script_dir / "inference_engine_pb2.pyi",
            ]:
                if generated_file.exists():
                    content = generated_file.read_text()
                    if "# mypy: ignore-errors" not in content.split("\n")[0:5]:
                        content = "# mypy: ignore-errors\n" + content
                    # Use relative import so the module works when loaded as inference_grpc.*
                    if generated_file.name == "inference_engine_pb2_grpc.py":
                        content = content.replace(
                            "import inference_engine_pb2 as inference__engine__pb2",
                            "from . import inference_engine_pb2 as inference__engine__pb2",
                        )
                    generated_file.write_text(content)

            print("✓ Protobuf compilation successful!")
            print(f"  Generated: {script_dir / 'inference_engine_pb2.py'}")
            print(f"  Generated: {script_dir / 'inference_engine_pb2_grpc.py'}")
            print(f"  Generated: {script_dir / 'inference_engine_pb2.pyi'} (type stubs)")
            return 0
        else:
            print(f"Error: protoc returned {result}")
            return result

    except ImportError:
        print("Error: grpcio-tools not installed")
        print("Install with: pip install grpcio-tools")
        return 1
    except Exception as e:
        print(f"Error during compilation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(compile_protos())
