#!/bin/bash

# CUDA 训练项目构建脚本
# 用法: ./compile.sh [选项]
# 选项:
#   -d, --debug      Debug 模式构建
#   -r, --release    Release 模式构建（默认）
#   -c, --clean      清理构建目录
#   -j, --jobs N     并行编译任务数（默认：自动检测）
#   -h, --help       显示帮助信息

set -e  # 遇到错误立即退出

# 默认配置
BUILD_TYPE="Release"
CLEAN=false
JOBS=$(nproc 2>/dev/null || echo 4)  # 自动检测 CPU 核心数，默认 4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "CUDA 训练项目构建脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -d, --debug       Debug 模式构建"
            echo "  -r, --release     Release 模式构建（默认）"
            echo "  -c, --clean       清理构建目录"
            echo "  -j, --jobs N      并行编译任务数（默认：自动检测）"
            echo "  -h, --help        显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                 # Release 模式构建"
            echo "  $0 -d              # Debug 模式构建"
            echo "  $0 -c              # 清理构建目录"
            echo "  $0 -d -j 8         # Debug 模式，8 个并行任务"
            exit 0
            ;;
        *)
            echo "错误: 未知选项 '$1'"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 清理构建目录
if [ "$CLEAN" = true ]; then
    echo "清理构建目录: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    echo "清理完成"
    exit 0
fi

# 创建构建目录
if [ ! -d "${BUILD_DIR}" ]; then
    echo "创建构建目录: ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
fi

# 进入构建目录
cd "${BUILD_DIR}"

# 运行 CMake 配置
echo "=========================================="
echo "配置 CMake (${BUILD_TYPE} 模式)"
echo "=========================================="
cmake .. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

# 编译
echo ""
echo "=========================================="
echo "开始编译 (${JOBS} 个并行任务)"
echo "=========================================="
cmake --build . -j "${JOBS}"

# 显示构建结果
echo ""
echo "=========================================="
echo "构建完成！"
echo "=========================================="
echo "构建类型: ${BUILD_TYPE}"
echo "可执行文件位置: ${BUILD_DIR}/bin/"
echo ""
echo "可执行文件列表:"
ls -lh "${BUILD_DIR}/bin/" 2>/dev/null || echo "  (无)"
echo ""
