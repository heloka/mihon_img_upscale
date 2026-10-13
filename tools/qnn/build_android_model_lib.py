#!/usr/bin/env python3
"""Build a weighted QNN Android model library from QAIRT converter output."""

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def build_model_library(
    qnn_sdk: Path,
    ndk: Path,
    model_cpp: Path,
    model_bin: Path,
    output: Path,
    android_api: int = 26,
) -> None:
    qnn_sdk = qnn_sdk.resolve()
    ndk = ndk.resolve()
    model_cpp = model_cpp.resolve()
    model_bin = model_bin.resolve()
    output = output.resolve()

    jni_dir = qnn_sdk / "share" / "QNN" / "converter" / "jni"
    qnn_headers = qnn_sdk / "include" / "QNN"
    pal_source = Path(__file__).with_name("android_qnn_model_pal.cpp").resolve()
    clang = ndk / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "bin" / "clang++.exe"
    objcopy = ndk / "toolchains" / "llvm" / "prebuilt" / "windows-x86_64" / "bin" / "llvm-objcopy.exe"
    required = [
        jni_dir / "QnnModel.cpp",
        jni_dir / "QnnWrapperUtils.cpp",
        jni_dir / "QnnModel.hpp",
        jni_dir / "QnnWrapperUtils.hpp",
        qnn_headers / "QnnInterface.h",
        pal_source,
        clang,
        objcopy,
        model_cpp,
        model_bin,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Android QNN model-library build inputs:\n" + "\n".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qnn-model-lib-") as temporary:
        build_dir = Path(temporary)
        binary_dir = build_dir / "obj" / "binary"
        binary_dir.mkdir(parents=True)

        with tarfile.open(model_bin, "r:*") as archive:
            raw_members = []
            for member in archive.getmembers():
                archive_path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or archive_path.is_absolute()
                    or ".." in archive_path.parts
                    or archive_path.suffix != ".raw"
                    or len(archive_path.parts) != 1
                ):
                    raise ValueError(f"Unexpected file in QNN model weight archive: {member.name}")
                raw_members.append(member)
            if not raw_members:
                raise ValueError(f"QNN weight archive has no .raw tensors: {model_bin}")
            for member in raw_members:
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unable to read weight tensor: {member.name}")
                with source, (binary_dir / member.name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)

        binary_objects = []
        for raw_file in sorted(binary_dir.glob("*.raw")):
            object_file = raw_file.with_suffix(raw_file.suffix + ".o")
            subprocess.run(
                [
                    str(objcopy),
                    "-I",
                    "binary",
                    "-O",
                    "elf64-littleaarch64",
                    "-B",
                    "aarch64",
                    raw_file.relative_to(build_dir).as_posix(),
                    object_file.relative_to(build_dir).as_posix(),
                ],
                cwd=build_dir,
                check=True,
            )
            binary_objects.append(object_file)

        candidate = build_dir / output.name
        compile_command = [
            str(clang),
            f"--target=aarch64-linux-android{android_api}",
            "-std=c++11",
            "-O3",
            "-fPIC",
            "-fvisibility=hidden",
            "-Wno-write-strings",
            "-Wno-c99-designator",
            '-DQNN_API=__attribute__((visibility("default")))',
            "-static-libstdc++",
            "-shared",
            "-Wl,-z,max-page-size=16384",
            "-I",
            str(qnn_headers),
            "-I",
            str(jni_dir),
            "-I",
            str(model_cpp.parent),
            str(jni_dir / "QnnModel.cpp"),
            str(jni_dir / "QnnWrapperUtils.cpp"),
            str(pal_source),
            str(model_cpp),
            *(str(path) for path in binary_objects),
            "-ldl",
            "-o",
            str(candidate),
        ]
        result = subprocess.run(compile_command, cwd=build_dir, capture_output=True, text=True)
        if result.returncode != 0:
            diagnostic = (result.stdout + "\n" + result.stderr).splitlines()
            raise RuntimeError(
                "Android model-library link failed:\n" + "\n".join(diagnostic[-60:]),
            )
        shutil.copy2(candidate, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qnn-sdk", type=Path, required=True)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--cpp", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--android-api", type=int, default=26)
    args = parser.parse_args()
    build_model_library(
        args.qnn_sdk,
        args.ndk,
        args.cpp,
        args.bin,
        args.output,
        args.android_api,
    )
    print(f"Built {args.output.resolve()}")


if __name__ == "__main__":
    main()
