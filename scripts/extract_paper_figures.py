#!/usr/bin/env python3
"""
extract_paper_figures.py

论文 PDF 图表提取和质量检查工具 (Python 版本)

功能:
    - 使用 pdfimages 优先提取原始图片，失败时自动降级到 pdftoppm
    - 自动格式转换 (PPM -> PNG)
    - 质量评分 (bytes/pixel)
    - 分辨率验证
    - 自动命名和文件验证
    - 详细的日志和错误报告

使用方式:
    python3 extract_paper_figures.py \\
        --pdf /path/to/paper.pdf \\
        --pages 2,6,7 \\
        --output-dir ./figures \\
        [--quality-threshold 0.5] \\
        [--min-width 600] \\
        [--max-size 500000]

示例:
    # 从 NormFormer 论文提取第2、6、7页的图表
    python3 extract_paper_figures.py \\
        --pdf /tmp/normformer.pdf \\
        --pages 2,6,7 \\
        --output-dir ./content/blog/posts/014_normformer_paper_review

依赖:
    - pdfimages (poppler 工具集)
    - pdftoppm (poppler 工具集)
    - convert (ImageMagick)
    - identify (ImageMagick)
    - Python 3.6+
    - Pillow (pip install Pillow)
"""

import argparse
import subprocess
import os
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import shutil
from dataclasses import dataclass, asdict
from enum import Enum


class QualityRating(Enum):
    """质量评级枚举"""
    EXCELLENT = "⭐⭐⭐⭐⭐ 优秀 (极好压缩)"
    VERY_GOOD = "⭐⭐⭐⭐ 很好 (很好压缩)"
    GOOD = "⭐⭐⭐ 良好 (可接受)"
    FAIR = "⭐⭐ 一般 (需要优化)"
    POOR = "⭐ 差 (必须优化)"


@dataclass
class ImageInfo:
    """图片信息数据类"""
    filename: str
    width: int
    height: int
    size_bytes: int
    bpp: float
    quality_rating: QualityRating
    checks: List[str]
    is_valid: bool


class ColorLogger:
    """带色彩的日志输出"""

    COLORS = {
        'INFO': '\033[0;34m',      # 蓝色
        'SUCCESS': '\033[0;32m',   # 绿色
        'WARNING': '\033[1;33m',   # 黄色
        'ERROR': '\033[0;31m',     # 红色
        'CYAN': '\033[0;36m',      # 青色
        'RESET': '\033[0m'         # 重置
    }

    @classmethod
    def _log(cls, level: str, message: str):
        """内部日志方法"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        color = cls.COLORS.get(level, '')
        reset = cls.COLORS['RESET']

        if level == 'SUCCESS':
            prefix = f"{color}[✓]{reset}"
        else:
            prefix = f"{color}[{level}]{reset}"

        print(f"{prefix} {timestamp} {message}")

        if level == 'ERROR':
            print(message, file=sys.stderr)

    @classmethod
    def info(cls, message: str):
        """信息日志"""
        cls._log('INFO', message)

    @classmethod
    def success(cls, message: str):
        """成功日志"""
        cls._log('SUCCESS', message)

    @classmethod
    def warning(cls, message: str):
        """警告日志"""
        cls._log('WARNING', message)

    @classmethod
    def error(cls, message: str):
        """错误日志"""
        cls._log('ERROR', message)


class ToolChecker:
    """工具检查"""

    REQUIRED_TOOLS = ['pdfimages', 'pdftoppm', 'convert', 'identify', 'pdfinfo']

    @classmethod
    def check_all(cls) -> bool:
        """检查所有必需工具"""
        missing = []

        for tool in cls.REQUIRED_TOOLS:
            if not shutil.which(tool):
                missing.append(tool)

        if missing:
            ColorLogger.error(f"缺少必需工具: {', '.join(missing)}")
            print("\n安装说明:")
            print("  macOS:  brew install imagemagick poppler")
            print("  Ubuntu: sudo apt-get install imagemagick poppler-utils")
            print("  CentOS: sudo yum install ImageMagick poppler-utils")
            return False

        ColorLogger.success("所有必需工具已就绪")
        return True


class PDFProcessor:
    """PDF 处理器"""

    def __init__(self, pdf_file: str, temp_dir: str, verbose: bool = False):
        self.pdf_file = pdf_file
        self.temp_dir = temp_dir
        self.verbose = verbose

    def get_pdf_info(self) -> Dict[str, str]:
        """获取 PDF 信息"""
        try:
            result = subprocess.run(
                ['pdfinfo', self.pdf_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()

            return info
        except Exception as e:
            ColorLogger.warning(f"无法获取 PDF 信息: {e}")
            return {}

    def extract_with_pdfimages(self, page: int, output_prefix: str) -> bool:
        """使用 pdfimages 提取图片 (优先方案)"""
        try:
            if self.verbose:
                ColorLogger.info(f"尝试 pdfimages 提取 (页 {page})...")

            result = subprocess.run(
                ['pdfimages', '-f', str(page), '-l', str(page),
                 self.pdf_file, output_prefix],
                capture_output=True,
                timeout=30
            )

            # 检查是否生成了文件
            temp_files = list(Path(self.temp_dir).glob(f'{Path(output_prefix).name}*'))
            if temp_files:
                if self.verbose:
                    ColorLogger.info(f"pdfimages 成功生成 {len(temp_files)} 个文件")
                return True

            if self.verbose:
                ColorLogger.warning("pdfimages 没有生成输出文件")
            return False

        except subprocess.TimeoutExpired:
            ColorLogger.error("pdfimages 执行超时")
            return False
        except Exception as e:
            if self.verbose:
                ColorLogger.warning(f"pdfimages 执行失败: {e}")
            return False

    def extract_with_pdftoppm(self, page: int, output_prefix: str) -> bool:
        """使用 pdftoppm 降级提取"""
        try:
            if self.verbose:
                ColorLogger.warning(f"降级到 pdftoppm (页 {page})...")

            result = subprocess.run(
                ['pdftoppm', '-f', str(page), '-l', str(page),
                 '-png', '-r', '150', self.pdf_file, output_prefix],
                capture_output=True,
                timeout=30
            )

            # 检查是否生成了文件
            temp_files = list(Path(self.temp_dir).glob(f'{Path(output_prefix).name}*'))
            if temp_files:
                if self.verbose:
                    ColorLogger.info(f"pdftoppm 成功生成 {len(temp_files)} 个文件")
                return True

            if self.verbose:
                ColorLogger.error("pdftoppm 执行失败")
            return False

        except subprocess.TimeoutExpired:
            ColorLogger.error("pdftoppm 执行超时")
            return False
        except Exception as e:
            if self.verbose:
                ColorLogger.warning(f"pdftoppm 执行失败: {e}")
            return False

    def extract_page(self, page: int, figure_num: int) -> Optional[str]:
        """提取单个页面"""
        ColorLogger.info(f"处理页面 #{page} (Figure {figure_num})...")

        temp_prefix = os.path.join(self.temp_dir, f'page_{page}_')

        # 尝试 pdfimages (优先)
        source_file = None

        if self.extract_with_pdfimages(page, temp_prefix):
            # 查找生成的文件
            candidate_files = list(Path(self.temp_dir).glob(f'page_{page}_*'))

            if candidate_files:
                for file in candidate_files:
                    if file.suffix in ['.png', '.jpg', '.ppm']:
                        if file.suffix == '.ppm':
                            source_file = self._convert_ppm_to_png(str(file))
                        else:
                            source_file = str(file)
                        break

        # 如果 pdfimages 失败或未找到文件，降级到 pdftoppm
        if not source_file:
            if self.extract_with_pdftoppm(page, temp_prefix):
                candidate_files = list(Path(self.temp_dir).glob(f'page_{page}_*'))
                if candidate_files:
                    source_file = str(candidate_files[0])

        if not source_file or not os.path.exists(source_file):
            ColorLogger.error(f"页面 {page} 提取失败，跳过")
            return None

        return source_file

    def _convert_ppm_to_png(self, ppm_file: str) -> Optional[str]:
        """PPM 转 PNG"""
        try:
            png_file = ppm_file.replace('.ppm', '.png')

            result = subprocess.run(
                ['convert', ppm_file, png_file],
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0 and os.path.exists(png_file):
                os.remove(ppm_file)
                return png_file

            return None
        except Exception as e:
            ColorLogger.error(f"PPM 转换失败: {e}")
            return None


class ImageValidator:
    """图片验证器"""

    def __init__(self, min_width: int = 600, min_height: int = 400,
                 max_size: int = 500000, quality_threshold: float = 0.5):
        self.min_width = min_width
        self.min_height = min_height
        self.max_size = max_size
        self.quality_threshold = quality_threshold

    def get_image_info(self, file_path: str) -> Optional[Tuple[int, int, int]]:
        """获取图片信息 (宽, 高, 大小)"""
        try:
            # 使用 identify 获取图片信息
            result = subprocess.run(
                ['identify', file_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return None

            # 解析输出: filename PNG 1275x900 ...
            parts = result.stdout.split()
            if len(parts) < 3:
                return None

            dimensions = parts[2].split('x')
            if len(dimensions) != 2:
                return None

            width = int(dimensions[0])
            height = int(dimensions[1])

            # 获取文件大小
            size_bytes = os.path.getsize(file_path)

            return width, height, size_bytes

        except Exception as e:
            ColorLogger.error(f"无法获取图片信息 {file_path}: {e}")
            return None

    def calculate_bpp(self, width: int, height: int, size_bytes: int) -> float:
        """计算 bytes per pixel"""
        pixels = width * height
        if pixels == 0:
            return 0.0
        return size_bytes / pixels

    def rate_quality(self, bpp: float) -> QualityRating:
        """评级质量"""
        if bpp < 0.15:
            return QualityRating.EXCELLENT
        elif bpp < 0.25:
            return QualityRating.VERY_GOOD
        elif bpp < self.quality_threshold:
            return QualityRating.GOOD
        elif bpp < 1.0:
            return QualityRating.FAIR
        else:
            return QualityRating.POOR

    def validate(self, file_path: str, figure_num: int) -> ImageInfo:
        """验证图片"""
        img_info_tuple = self.get_image_info(file_path)

        if not img_info_tuple:
            return ImageInfo(
                filename=os.path.basename(file_path),
                width=0,
                height=0,
                size_bytes=0,
                bpp=0,
                quality_rating=QualityRating.POOR,
                checks=["❌ 无法读取图片信息"],
                is_valid=False
            )

        width, height, size_bytes = img_info_tuple
        bpp = self.calculate_bpp(width, height, size_bytes)
        quality_rating = self.rate_quality(bpp)

        checks = []
        is_valid = True

        # 宽度检查
        if width < self.min_width:
            checks.append(f"❌ 宽度过小: {width}px < {self.min_width}px")
            is_valid = False
        else:
            checks.append(f"✓ 宽度: {width}px")

        # 高度检查
        if height < self.min_height:
            checks.append(f"❌ 高度过小: {height}px < {self.min_height}px")
            is_valid = False
        else:
            checks.append(f"✓ 高度: {height}px")

        # 大小检查
        size_kb = size_bytes / 1024
        if size_bytes > self.max_size:
            checks.append(f"❌ 文件过大: {size_kb:.1f}KB > {self.max_size / 1024:.0f}KB")
            is_valid = False
        else:
            checks.append(f"✓ 文件大小: {size_kb:.1f}KB")

        # bytes/pixel 检查
        if bpp > self.quality_threshold:
            checks.append(f"⚠ 压缩率: {bpp:.4f} bytes/pixel (>{self.quality_threshold} 阈值)")
        else:
            checks.append(f"✓ 压缩率: {bpp:.4f} bytes/pixel")

        return ImageInfo(
            filename=os.path.basename(file_path),
            width=width,
            height=height,
            size_bytes=size_bytes,
            bpp=bpp,
            quality_rating=quality_rating,
            checks=checks,
            is_valid=is_valid
        )


class ReportGenerator:
    """报告生成器"""

    @staticmethod
    def generate(
        output_dir: str,
        pdf_file: str,
        pages: List[int],
        quality_threshold: float,
        min_width: int,
        min_height: int,
        max_size: int,
        image_info_list: List[ImageInfo]
    ) -> str:
        """生成 Markdown 质量报告"""

        report_file = os.path.join(output_dir, 'figures_quality_report.md')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 论文图表提取质量报告\n\n")

            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**输入 PDF**: {pdf_file}\n")
            f.write(f"**输出目录**: {output_dir}\n\n")

            # 质量标准
            f.write("## 质量标准\n\n")
            f.write("| 指标 | 标准 |\n")
            f.write("|------|------|\n")
            f.write(f"| 最小宽度 | {min_width}px |\n")
            f.write(f"| 最小高度 | {min_height}px |\n")
            f.write(f"| 最大文件大小 | {max_size / 1024:.0f}KB |\n")
            f.write(f"| bytes/pixel 阈值 | {quality_threshold} |\n\n")

            # 提取结果
            f.write("## 提取结果\n\n")

            if not image_info_list:
                f.write("⚠ 没有成功提取任何图表\n\n")
            else:
                for idx, info in enumerate(image_info_list, 1):
                    f.write(f"### Figure {idx}: {info.filename}\n\n")
                    f.write(f"- **尺寸**: {info.width}×{info.height}px\n")
                    f.write(f"- **文件大小**: {info.size_bytes / 1024:.1f}KB\n")
                    f.write(f"- **bytes/pixel**: {info.bpp:.4f}\n")
                    f.write(f"- **质量评级**: {info.quality_rating.value}\n")
                    f.write(f"- **状态**: {'✓ 通过' if info.is_valid else '❌ 不通过'}\n\n")

            # 建议
            f.write("## 建议\n\n")
            f.write("1. **图表质量**: 所有提取的图表都应满足上述标准\n")
            f.write("2. **进一步优化**: 如有评级低于 ⭐⭐⭐ 的图表，可考虑调整提取参数或在图像编辑器中手工优化\n")
            f.write("3. **集成**: 将生成的图表复制到博客目录后，更新 Markdown 文件中的引用\n\n")

            # 调试信息
            f.write("## 调试信息\n\n")
            f.write("```\n")
            f.write(f"PDF 文件: {pdf_file}\n")
            f.write(f"输出目录: {output_dir}\n")
            f.write(f"提取的页码: {','.join(map(str, pages))}\n")
            f.write(f"质量标准: bytes/pixel <= {quality_threshold}\n")
            f.write(f"分辨率标准: {min_width}×{min_height}px 及以上\n")
            f.write("```\n")

        return report_file


class PaperFigureExtractor:
    """主提取器类"""

    def __init__(self, args):
        self.pdf_file = args.pdf
        self.pages = [int(p.strip()) for p in args.pages.split(',')]
        self.output_dir = args.output_dir
        self.quality_threshold = args.quality_threshold
        self.min_width = args.min_width
        self.min_height = args.min_height
        self.max_size = args.max_size
        self.enable_report = not args.no_report
        self.verbose = args.verbose
        self.temp_dir = tempfile.mkdtemp(prefix='pdf_extract_')

    def run(self) -> int:
        """运行提取流程"""

        ColorLogger.info("=" * 50)
        ColorLogger.info("论文 PDF 图表提取工具")
        ColorLogger.info("=" * 50)
        print()

        # 验证参数
        if not self._validate_parameters():
            return 1

        # 检查工具
        if not ToolChecker.check_all():
            return 1

        # 初始化
        self._initialize()

        print()

        # 提取和验证
        processor = PDFProcessor(self.pdf_file, self.temp_dir, self.verbose)
        validator = ImageValidator(
            self.min_width,
            self.min_height,
            self.max_size,
            self.quality_threshold
        )

        success_count = 0
        fail_count = 0
        image_info_list = []

        for idx, page in enumerate(self.pages, 1):
            source_file = processor.extract_page(page, idx)

            if source_file:
                target_file = os.path.join(self.output_dir, f'figure_{idx}.png')

                # 如果目标文件已存在，备份
                if os.path.exists(target_file):
                    backup_file = target_file.replace('.png', '.backup.png')
                    ColorLogger.warning(f"目标文件已存在，备份为: {backup_file}")
                    shutil.copy2(target_file, backup_file)

                # 复制文件
                shutil.copy2(source_file, target_file)
                ColorLogger.success(f"提取完成: {target_file}")

                # 验证
                image_info = validator.validate(target_file, idx)
                image_info_list.append(image_info)

                ColorLogger.info(f"Figure {idx} 验证结果:")
                for check in image_info.checks:
                    print(f"  {check}")
                print(f"  质量评级: {image_info.quality_rating.value}")

                if image_info.is_valid:
                    ColorLogger.success(f"Figure {idx} 验证通过")
                    success_count += 1
                else:
                    ColorLogger.warning(f"Figure {idx} 存在问题")
                    fail_count += 1
            else:
                fail_count += 1

            print()

        # 生成报告
        if self.enable_report:
            report_file = ReportGenerator.generate(
                self.output_dir,
                self.pdf_file,
                self.pages,
                self.quality_threshold,
                self.min_width,
                self.min_height,
                self.max_size,
                image_info_list
            )
            ColorLogger.success(f"质量报告已生成: {report_file}")

        # 清理临时文件
        shutil.rmtree(self.temp_dir)

        # 最终统计
        print()
        ColorLogger.info("=" * 50)
        ColorLogger.info("提取完成")
        ColorLogger.info("=" * 50)
        ColorLogger.success(f"成功: {success_count} / {len(self.pages)}")

        if fail_count > 0:
            ColorLogger.warning(f"失败: {fail_count} / {len(self.pages)}")
            return 1
        else:
            ColorLogger.success("所有图表已成功提取和验证")
            return 0

    def _validate_parameters(self) -> bool:
        """验证参数"""
        errors = []

        if not os.path.isfile(self.pdf_file):
            errors.append(f"PDF 文件不存在: {self.pdf_file}")

        if not self.pages:
            errors.append("页码列表为空")

        if errors:
            ColorLogger.error("参数验证失败:")
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return False

        return True

    def _initialize(self) -> bool:
        """初始化"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        ColorLogger.info(f"输出目录: {self.output_dir}")

        # 获取 PDF 信息
        processor = PDFProcessor(self.pdf_file, self.temp_dir, self.verbose)
        pdf_info = processor.get_pdf_info()

        ColorLogger.info(f"PDF 信息: {self.pdf_file}")
        if pdf_info.get('Pages'):
            ColorLogger.info(f"  - 页数: {pdf_info['Pages']}")
        if pdf_info.get('File size'):
            ColorLogger.info(f"  - 大小: {pdf_info['File size']}")

        ColorLogger.info(f"待提取页码: {','.join(map(str, self.pages))} (共 {len(self.pages)} 个)")

        return True


def main():
    """主函数"""

    parser = argparse.ArgumentParser(
        description="论文 PDF 图表提取和质量检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 NormFormer 论文提取第2、6、7页的图表
  python3 extract_paper_figures.py \\
    --pdf /tmp/normformer.pdf \\
    --pages 2,6,7 \\
    --output-dir ./content/blog/posts/014_normformer_paper_review

  # 自定义质量标准
  python3 extract_paper_figures.py \\
    --pdf paper.pdf \\
    --pages 1,3,4,5 \\
    --output-dir ./blog/posts/123_review/ \\
    --quality-threshold 0.3 \\
    --max-size 300000
        """
    )

    parser.add_argument('--pdf', required=True, help='输入 PDF 文件路径')
    parser.add_argument('--pages', required=True, help='要提取的页码，逗号分隔 (例: "2,6,7")')
    parser.add_argument('--output-dir', required=True, help='输出目录路径')
    parser.add_argument('--quality-threshold', type=float, default=0.5,
                        help='字节/像素质量阈值 (默认: 0.5)')
    parser.add_argument('--min-width', type=int, default=600,
                        help='最小宽度像素数 (默认: 600)')
    parser.add_argument('--min-height', type=int, default=400,
                        help='最小高度像素数 (默认: 400)')
    parser.add_argument('--max-size', type=int, default=500000,
                        help='最大文件大小字节 (默认: 500000)')
    parser.add_argument('--no-report', action='store_true',
                        help='不生成质量报告')
    parser.add_argument('--verbose', action='store_true',
                        help='输出详细调试信息')

    args = parser.parse_args()

    extractor = PaperFigureExtractor(args)
    return extractor.run()


if __name__ == '__main__':
    sys.exit(main())

