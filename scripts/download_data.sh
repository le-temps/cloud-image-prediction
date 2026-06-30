#!/bin/bash
# 下载 SKIPP'D 数据集
# 数据集存放在 data/skippd/

set -e

DATA_DIR="./data/skippd"
mkdir -p $DATA_DIR

echo "========================================="
echo "Downloading SKIPP'D Dataset"
echo "Source: https://zenodo.org/records/7560637"
echo "========================================="

# 方式1: 使用 zenodo-get (推荐)
if command -v zenodo_get &> /dev/null; then
    echo "Using zenodo_get..."
    cd $DATA_DIR
    zenodo_get 7560637
    cd -
else
    echo "zenodo_get not found. Install with: pip install zenodo_get"
    echo ""
    echo "Alternative: Manual download from:"
    echo "  https://zenodo.org/records/7560637"
    echo ""
    echo "Or use wget (if direct link available):"
    echo "  wget -P $DATA_DIR https://zenodo.org/records/7560637/files/SKIPPD.zip"
    echo ""
    echo "After download, extract to $DATA_DIR/images/"
fi

echo ""
echo "========================================="
echo "Dataset setup complete."
echo "Expected structure:"
echo "  data/skippd/"
echo "  ├── images/"
echo "  │   ├── *.jpg"
echo "  │   └── ..."
echo "  └── metadata.csv"
echo "========================================="
