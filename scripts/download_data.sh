#!/bin/bash
# 数据集下载脚本
# 两个数据集:
#   1. SKIPP'D Raw (2048x2048) - 斯坦福大学, 主实验
#   2. NREL TSI-880 (1024x1024) - 美国NREL, 泛化验证
set -e

echo "========================================="
echo "  Cloud Image Prediction - Data Download"
echo "========================================="

#############################################
# 1. SKIPP'D Raw Dataset (Stanford)
#############################################
echo ""
echo "[1/2] SKIPP'D Raw Dataset (2048x2048, Stanford)"
echo "    原始分辨率: 2048x2048 鱼眼相机"
echo "    采样频率: 1min"
echo "    时间: 2017.03 - 2019.12"
echo ""

SKIPPD_DIR="./data/skippd_raw"
mkdir -p $SKIPPD_DIR/images $SKIPPD_DIR/pv

echo "下载方式 (选择一种):"
echo ""
echo "  方式A - 通过 Stanford Digital Repository 下载 Raw 数据:"
echo "    浏览器打开: https://purl.stanford.edu/sm043zf7254"
echo "    下载 {Year}_{Month}_images_raw.tar 文件 (每月约7GB)"
echo "    解压到 $SKIPPD_DIR/images/"
echo ""
echo "  方式B - 通过 wget 下载 (需科学上网或海外服务器):"
echo "    # 2019年数据 (推荐先下载一年做实验)"
echo "    wget -P $SKIPPD_DIR/ 'https://stacks.stanford.edu/file/druid:sm043zf7254/2019_01_images_raw.tar'"
echo "    wget -P $SKIPPD_DIR/ 'https://stacks.stanford.edu/file/druid:sm043zf7254/2019_02_images_raw.tar'"
echo "    # ... (每月一个tar包)"
echo ""
echo "  方式C - 只下载 Benchmark 数据 (64x64, 用于快速验证):"
echo "    浏览器打开: https://purl.stanford.edu/dj417rh1007"
echo "    下载 2017_2019_images_pv_processed.hdf5"
echo "    放到 $SKIPPD_DIR/"
echo ""
echo "  方式D - PV功率数据:"
echo "    同 Raw 页面下载 {Year}_pv_raw.csv"
echo "    放到 $SKIPPD_DIR/pv/"
echo ""

# 解压示例 (下载后执行)
cat > $SKIPPD_DIR/extract.sh << 'EXTRACT'
#!/bin/bash
# 解压所有 tar 包到 images 目录
for f in *.tar; do
    echo "Extracting $f..."
    tar -xf "$f" -C images/
done
echo "Done! Total images:"
find images/ -name "*.jpg" | wc -l
EXTRACT
chmod +x $SKIPPD_DIR/extract.sh

#############################################
# 2. NREL TSI-880 Dataset
#############################################
echo ""
echo "[2/2] NREL TSI-880 Dataset (1024x1024)"
echo "    设备: TSI-880 全天空成像仪"
echo "    来源: NREL Solar Radiation Research Laboratory"
echo "    位置: Golden, Colorado"
echo ""

NREL_DIR="./data/nrel_tsi"
mkdir -p $NREL_DIR/images

echo "下载方式:"
echo ""
echo "  NREL Measurement & Instrumentation Data Center:"
echo "    https://midcdmz.nrel.gov/"
echo ""
echo "  步骤:"
echo "    1. 访问 https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS"
echo "    2. 选择 'Total Sky Imager' 数据"
echo "    3. 选择时间范围 (建议: 2019-2020, 至少6个月)"
echo "    4. 下载图像文件，放到 $NREL_DIR/images/"
echo ""
echo "  或直接访问 TSI 图像存档:"
echo "    https://midcdmz.nrel.gov/tsi/"
echo ""

#############################################
# 数据预处理脚本
#############################################
echo ""
echo "========================================="
echo "下载完成后，运行预处理:"
echo "  python scripts/preprocess_data.py --dataset skippd_raw --root $SKIPPD_DIR"
echo "  python scripts/preprocess_data.py --dataset nrel_tsi --root $NREL_DIR"
echo "========================================="
