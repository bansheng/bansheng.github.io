#!/bin/bash
#
# extract_paper_figures.sh
# 论文 PDF 图表提取和质量检查工具
#
# 功能:
#   - 使用 pdfimages 优先提取原始图片，失败时自动降级到 pdftoppm
#   - 自动格式转换 (PPM -> PNG)
#   - 质量评分 (bytes/pixel)
#   - 分辨率验证
#   - 自动命名和文件验证
#   - 详细的日志和错误报告
#
# 使用方式:
#   ./extract_paper_figures.sh \
#     --pdf /path/to/paper.pdf \
#     --pages "2,6,7" \
#     --output-dir ./figures \
#     [--quality-threshold 0.5] \
#     [--min-width 600] \
#     [--max-size 500000]
#
# 示例:
#   # 从 NormFormer 论文提取第2、6、7页的图表
#   ./extract_paper_figures.sh \
#     --pdf /tmp/normformer.pdf \
#     --pages "2,6,7" \
#     --output-dir ./content/blog/posts/014_normformer_paper_review
#
# 依赖:
#   - pdfimages (poppler 工具集)
#   - pdftoppm (poppler 工具集)
#   - convert (ImageMagick)
#   - identify (ImageMagick)
#   - bc (数学计算)
#

set -euo pipefail

# ============================================================================
# 颜色输出和日志
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

# ============================================================================
# 默认参数
# ============================================================================

PDF_FILE=""
PAGES=""
OUTPUT_DIR=""
QUALITY_THRESHOLD=0.5
MIN_WIDTH=600
MIN_HEIGHT=400
MAX_SIZE=500000  # 字节
ENABLE_REPORT=true
TEMP_DIR="/tmp/pdf_extract_$$"
REPORT_FILE=""
VERBOSE=false

# ============================================================================
# 参数解析
# ============================================================================

usage() {
    cat << 'EOF'
使用方式：extract_paper_figures.sh [选项]

必需参数：
  --pdf PATH              输入 PDF 文件路径
  --pages PAGES           要提取的页码，逗号分隔 (例: "2,6,7")
  --output-dir DIR        输出目录路径

可选参数：
  --quality-threshold N   字节/像素质量阈值 (默认: 0.5)
  --min-width N           最小宽度像素数 (默认: 600)
  --min-height N          最小高度像素数 (默认: 400)
  --max-size N            最大文件大小字节 (默认: 500000)
  --no-report             不生成质量报告
  --verbose               输出详细调试信息
  --help                  显示此帮助信息

示例：
  ./extract_paper_figures.sh \
    --pdf /tmp/normformer.pdf \
    --pages "2,6,7" \
    --output-dir ./figures

  ./extract_paper_figures.sh \
    --pdf paper.pdf \
    --pages "1,3,4,5" \
    --output-dir ./blog/posts/123_review/ \
    --quality-threshold 0.3 \
    --max-size 300000

EOF
    exit 1
}

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --pdf)
            PDF_FILE="$2"
            shift 2
            ;;
        --pages)
            PAGES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quality-threshold)
            QUALITY_THRESHOLD="$2"
            shift 2
            ;;
        --min-width)
            MIN_WIDTH="$2"
            shift 2
            ;;
        --min-height)
            MIN_HEIGHT="$2"
            shift 2
            ;;
        --max-size)
            MAX_SIZE="$2"
            shift 2
            ;;
        --no-report)
            ENABLE_REPORT=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            log_error "未知参数: $1"
            usage
            ;;
    esac
done

# ============================================================================
# 参数验证
# ============================================================================

validate_parameters() {
    local errors=()

    if [[ -z "$PDF_FILE" ]]; then
        errors+=("缺少 --pdf 参数")
    elif [[ ! -f "$PDF_FILE" ]]; then
        errors+=("PDF 文件不存在: $PDF_FILE")
    fi

    if [[ -z "$PAGES" ]]; then
        errors+=("缺少 --pages 参数")
    fi

    if [[ -z "$OUTPUT_DIR" ]]; then
        errors+=("缺少 --output-dir 参数")
    fi

    if [[ ${#errors[@]} -gt 0 ]]; then
        log_error "参数验证失败:"
        printf '%s\n' "${errors[@]}" | sed 's/^/  - /'
        echo ""
        usage
    fi
}

# ============================================================================
# 工具检查
# ============================================================================

check_tools() {
    local missing_tools=()

    for tool in pdfimages pdftoppm convert identify bc; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "缺少必需工具: ${missing_tools[*]}"
        echo ""
        echo "安装说明："
        echo "  macOS:  brew install imagemagick poppler"
        echo "  Ubuntu: sudo apt-get install imagemagick poppler-utils"
        echo "  CentOS: sudo yum install ImageMagick poppler-utils"
        exit 1
    fi

    log_success "所有必需工具已就绪"
}

# ============================================================================
# 初始化
# ============================================================================

initialize() {
    # 创建临时目录
    mkdir -p "$TEMP_DIR"
    log_info "临时目录: $TEMP_DIR"

    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    log_info "输出目录: $OUTPUT_DIR"

    # 设置报告文件路径
    REPORT_FILE="$OUTPUT_DIR/figures_quality_report.md"

    # 获取 PDF 信息
    local pdf_size_bytes=$(stat -f%z "$PDF_FILE" 2>/dev/null || stat -c%s "$PDF_FILE")
    local pdf_size_mb=$(echo "scale=2; $pdf_size_bytes / 1024 / 1024" | bc)
    local pdf_pages=$(pdfinfo "$PDF_FILE" 2>/dev/null | grep Pages | awk '{print $2}' || echo "未知")

    log_info "PDF 信息: $PDF_FILE"
    log_info "  - 大小: ${pdf_size_mb} MB"
    log_info "  - 页数: $pdf_pages"
}

# ============================================================================
# 质量评分函数
# ============================================================================

# 计算 bytes/pixel 比率
calculate_bytes_per_pixel() {
    local file=$1
    local width=$2
    local height=$3

    local size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
    local pixels=$((width * height))

    if [[ $pixels -eq 0 ]]; then
        echo "0"
        return
    fi

    echo "scale=4; $size_bytes / $pixels" | bc
}

# 评级函数 (基于 bytes/pixel)
rate_quality() {
    local bpp=$1
    local threshold=$2

    if (( $(echo "$bpp < 0.15" | bc -l) )); then
        echo "⭐⭐⭐⭐⭐ 优秀 (极好压缩)"
    elif (( $(echo "$bpp < 0.25" | bc -l) )); then
        echo "⭐⭐⭐⭐ 很好 (很好压缩)"
    elif (( $(echo "$bpp < $threshold" | bc -l) )); then
        echo "⭐⭐⭐ 良好 (可接受)"
    elif (( $(echo "$bpp < 1.0" | bc -l) )); then
        echo "⭐⭐ 一般 (需要优化)"
    else
        echo "⭐ 差 (必须优化)"
    fi
}

# ============================================================================
# 提取图片函数
# ============================================================================

# 使用 pdfimages 提取 (优先)
extract_with_pdfimages() {
    local pdf=$1
    local page=$2
    local output_prefix=$3

    [[ $VERBOSE == true ]] && log_info "尝试 pdfimages 提取 (页 $page)..."

    if pdfimages -f "$page" -l "$page" "$pdf" "$output_prefix" 2>/dev/null; then
        # 检查是否生成了文件
        if ls "${output_prefix}"* &> /dev/null; then
            return 0
        else
            [[ $VERBOSE == true ]] && log_warn "pdfimages 没有生成输出文件"
            return 1
        fi
    else
        [[ $VERBOSE == true ]] && log_warn "pdfimages 执行失败"
        return 1
    fi
}

# 使用 pdftoppm 降级提取
extract_with_pdftoppm() {
    local pdf=$1
    local page=$2
    local output_prefix=$3

    [[ $VERBOSE == true ]] && log_warn "降级到 pdftoppm (页 $page)..."

    if pdftoppm -f "$page" -l "$page" -png -r 150 "$pdf" "$output_prefix" 2>/dev/null; then
        return 0
    else
        [[ $VERBOSE == true ]] && log_error "pdftoppm 执行失败"
        return 1
    fi
}

# 转换 PPM 到 PNG
convert_ppm_to_png() {
    local ppm_file=$1
    local png_file=$2

    if [[ ! -f "$ppm_file" ]]; then
        return 1
    fi

    if convert "$ppm_file" "$png_file" 2>/dev/null; then
        rm -f "$ppm_file"
        return 0
    else
        log_error "PPM 转换失败: $ppm_file"
        return 1
    fi
}

# 提取单个页面
extract_page() {
    local page=$1
    local figure_num=$2

    log_info "处理页面 #$page (Figure $figure_num)..."

    local temp_prefix="$TEMP_DIR/page_${page}_"

    # 尝试 pdfimages (优先)
    if extract_with_pdfimages "$PDF_FILE" "$page" "$temp_prefix"; then
        [[ $VERBOSE == true ]] && log_info "pdfimages 成功提取"

        # 处理提取的文件
        local source_file=""
        if [[ -f "${temp_prefix}000.png" ]]; then
            source_file="${temp_prefix}000.png"
        elif [[ -f "${temp_prefix}000.ppm" ]]; then
            convert_ppm_to_png "${temp_prefix}000.ppm" "${temp_prefix}000.png" || {
                log_warn "PPM 转换失败，尝试 pdftoppm..."
                extract_with_pdftoppm "$PDF_FILE" "$page" "$temp_prefix"
                source_file="${temp_prefix}-0${page}.png"
            }
            source_file="${temp_prefix}000.png"
        else
            log_warn "pdfimages 没有生成预期格式，尝试 pdftoppm..."
            extract_with_pdftoppm "$PDF_FILE" "$page" "$temp_prefix"
            source_file="${temp_prefix}-0${page}.png"
        fi
    else
        # 降级到 pdftoppm
        if extract_with_pdftoppm "$PDF_FILE" "$page" "$temp_prefix"; then
            source_file="${temp_prefix}-0${page}.png"
        else
            log_error "页面 $page 提取失败，跳过"
            return 1
        fi
    fi

    # 验证源文件存在
    if [[ ! -f "$source_file" ]]; then
        log_error "提取后找不到源文件: $source_file"
        return 1
    fi

    # 生成目标文件名
    local target_file="$OUTPUT_DIR/figure_${figure_num}.png"

    # 如果已存在，备份原文件
    if [[ -f "$target_file" ]]; then
        local backup_file="$OUTPUT_DIR/figure_${figure_num}.backup.png"
        log_warn "目标文件已存在，备份为: $backup_file"
        cp "$target_file" "$backup_file"
    fi

    # 复制文件
    cp "$source_file" "$target_file" || {
        log_error "复制文件失败: $source_file -> $target_file"
        return 1
    }

    log_success "提取完成: $target_file"

    # 返回目标文件路径用于验证
    echo "$target_file"
}

# ============================================================================
# 验证函数
# ============================================================================

validate_image() {
    local file=$1
    local figure_num=$2
    local width height size_bytes bpp quality_rating

    if [[ ! -f "$file" ]]; then
        log_error "文件验证失败：文件不存在 $file"
        return 1
    fi

    # 获取图片信息
    local img_info
    img_info=$(identify "$file" 2>/dev/null) || {
        log_error "无法读取图片信息: $file"
        return 1
    }

    # 解析宽高
    width=$(echo "$img_info" | awk '{print $3}' | cut -d'x' -f1)
    height=$(echo "$img_info" | awk '{print $3}' | cut -d'x' -f2)

    # 获取文件大小
    size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
    local size_kb=$(echo "scale=1; $size_bytes / 1024" | bc)

    # 计算 bytes/pixel
    bpp=$(calculate_bytes_per_pixel "$file" "$width" "$height")

    # 评级
    quality_rating=$(rate_quality "$bpp" "$QUALITY_THRESHOLD")

    # 验证检查
    local checks=()
    local pass=true

    # 宽度检查
    if [[ $width -lt $MIN_WIDTH ]]; then
        checks+=("❌ 宽度过小: ${width}px < ${MIN_WIDTH}px")
        pass=false
    else
        checks+=("✓ 宽度: ${width}px")
    fi

    # 高度检查
    if [[ $height -lt $MIN_HEIGHT ]]; then
        checks+=("❌ 高度过小: ${height}px < ${MIN_HEIGHT}px")
        pass=false
    else
        checks+=("✓ 高度: ${height}px")
    fi

    # 大小检查
    if [[ $size_bytes -gt $MAX_SIZE ]]; then
        checks+=("❌ 文件过大: ${size_kb}KB > $(echo "scale=0; $MAX_SIZE / 1024" | bc)KB")
        pass=false
    else
        checks+=("✓ 文件大小: ${size_kb}KB")
    fi

    # bytes/pixel 检查
    if (( $(echo "$bpp > $QUALITY_THRESHOLD" | bc -l) )); then
        checks+=("⚠ 压缩率: $bpp bytes/pixel (>$QUALITY_THRESHOLD 阈值)")
    else
        checks+=("✓ 压缩率: $bpp bytes/pixel")
    fi

    # 输出结果
    log_info "Figure $figure_num 验证结果:"
    printf '%s\n' "${checks[@]}" | sed 's/^/  /'
    echo -e "  质量评级: $quality_rating"

    if [[ $pass == false ]]; then
        log_warn "Figure $figure_num 存在问题"
        return 1
    else
        log_success "Figure $figure_num 验证通过"
        return 0
    fi
}

# ============================================================================
# 报告生成
# ============================================================================

generate_report() {
    local report_data=()

    {
        echo "# 论文图表提取质量报告"
        echo ""
        echo "**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "**输入 PDF**: $PDF_FILE"
        echo "**输出目录**: $OUTPUT_DIR"
        echo ""

        echo "## 质量标准"
        echo ""
        echo "| 指标 | 标准 |"
        echo "|------|------|"
        echo "| 最小宽度 | ${MIN_WIDTH}px |"
        echo "| 最小高度 | ${MIN_HEIGHT}px |"
        echo "| 最大文件大小 | $(echo "scale=0; $MAX_SIZE / 1024" | bc)KB |"
        echo "| bytes/pixel 阈值 | $QUALITY_THRESHOLD |"
        echo ""

        echo "## 提取结果"
        echo ""

        # 列举所有生成的文件
        local figure_count=0
        for file in "$OUTPUT_DIR"/figure_*.png; do
            if [[ -f "$file" ]]; then
                ((figure_count++))
                local basename=$(basename "$file")
                local size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
                local size_kb=$(echo "scale=1; $size_bytes / 1024" | bc)

                local img_info=$(identify "$file" 2>/dev/null)
                local width=$(echo "$img_info" | awk '{print $3}' | cut -d'x' -f1)
                local height=$(echo "$img_info" | awk '{print $3}' | cut -d'x' -f2)

                local bpp=$(calculate_bytes_per_pixel "$file" "$width" "$height")
                local rating=$(rate_quality "$bpp" "$QUALITY_THRESHOLD")

                echo "### $basename"
                echo ""
                echo "- **尺寸**: ${width}×${height}px"
                echo "- **文件大小**: ${size_kb}KB"
                echo "- **bytes/pixel**: $bpp"
                echo "- **质量评级**: $rating"
                echo ""
            fi
        done

        if [[ $figure_count -eq 0 ]]; then
            echo "⚠ 没有生成任何图表文件"
            echo ""
        fi

        echo "## 建议"
        echo ""
        echo "1. **图表质量**: 所有提取的图表都应满足上述标准"
        echo "2. **进一步优化**: 如有评级低于 ⭐⭐⭐ 的图表，可考虑调整提取参数或在图像编辑器中手工优化"
        echo "3. **集成**: 将生成的图表复制到博客目录后，更新 Markdown 文件中的引用"
        echo ""

        echo "## 调试信息"
        echo ""
        echo "\`\`\`"
        echo "PDF 文件: $PDF_FILE"
        echo "输出目录: $OUTPUT_DIR"
        echo "提取的页码: $PAGES"
        echo "质量标准: bytes/pixel <= $QUALITY_THRESHOLD"
        echo "分辨率标准: ${MIN_WIDTH}×${MIN_HEIGHT}px 及以上"
        echo "\`\`\`"

    } > "$REPORT_FILE"

    log_success "质量报告已生成: $REPORT_FILE"
}

# ============================================================================
# 主函数
# ============================================================================

main() {
    log_info "=========================================="
    log_info "论文 PDF 图表提取工具"
    log_info "=========================================="
    echo ""

    # 参数验证和初始化
    validate_parameters
    check_tools
    initialize

    echo ""

    # 解析页码
    IFS=',' read -ra page_array <<< "$PAGES"

    if [[ ${#page_array[@]} -eq 0 ]]; then
        log_error "页码解析失败"
        exit 1
    fi

    log_info "待提取页码: ${page_array[*]} (共 ${#page_array[@]} 个)"
    echo ""

    # 提取图片并验证
    local success_count=0
    local fail_count=0

    for i in "${!page_array[@]}"; do
        local page=${page_array[$i]}
        local figure_num=$((i + 1))

        if target_file=$(extract_page "$page" "$figure_num"); then
            if validate_image "$target_file" "$figure_num"; then
                ((success_count++))
            else
                ((fail_count++))
            fi
        else
            ((fail_count++))
        fi

        echo ""
    done

    # 生成报告
    if [[ $ENABLE_REPORT == true ]]; then
        generate_report
    fi

    # 清理临时文件
    rm -rf "$TEMP_DIR"

    # 最终统计
    echo ""
    log_info "=========================================="
    log_info "提取完成"
    log_info "=========================================="
    log_success "成功: $success_count / ${#page_array[@]}"

    if [[ $fail_count -gt 0 ]]; then
        log_warn "失败: $fail_count / ${#page_array[@]}"
        log_info "输出目录: $OUTPUT_DIR"
        log_info "质量报告: $REPORT_FILE"
        echo ""
        return 1
    else
        log_success "所有图表已成功提取和验证"
        log_info "输出目录: $OUTPUT_DIR"
        log_info "质量报告: $REPORT_FILE"
        echo ""
        return 0
    fi
}

# 运行主函数
main "$@"

