# backend/rag/excel_loader.py
"""
专门处理复杂合并单元格的Excel加载器 - 针对新格式（无图片路径）
处理格式：分区说明行 + 大类表头行 + 详细表头行 + 多行数据
"""
import pandas as pd
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document


class ComplexExhibitionExcelLoader:
    """复杂格式的展览数据Excel加载器（新格式版本）"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

        # 类别映射表
        self.category_map = {
            "工业设计类": "工业设计",
            "环境设计类": "环境设计",
            "艺术与科技类": "艺术与科技",
        }

    def load_all_sheets(self) -> List[Document]:
        """
        加载Excel文件的所有sheet
        """
        all_documents = []

        try:
            print(f"📚 加载文件: {self.file_name}")

            # 读取Excel文件
            excel_file = pd.ExcelFile(self.file_path, engine='openpyxl')
            sheet_names = excel_file.sheet_names

            print(f"📋 发现 {len(sheet_names)} 个sheet: {sheet_names}")

            # 处理每个sheet
            for sheet_name in sheet_names:
                print(f"\n  ── 处理: {sheet_name} ──")

                sheet_docs = self._process_complex_sheet(sheet_name, excel_file)
                all_documents.extend(sheet_docs)

                print(f"  ✅ 生成 {len(sheet_docs)} 个文档片段")

            print(f"\n📈 总计生成: {len(all_documents)} 个文档片段")

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()

        return all_documents

    def _process_complex_sheet(self, sheet_name: str, excel_file: pd.ExcelFile) -> List[Document]:
        """
        处理复杂格式的sheet（新格式）
        """
        documents = []

        try:
            # 1. 读取原始数据，不指定表头
            raw_df = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                header=None,  # 不指定表头
                dtype=str,
                keep_default_na=False,
                engine='openpyxl'
            )

            print(f"    原始数据形状: {raw_df.shape}")

            # 显示前几行数据用于调试
            print(f"\n    前6行数据预览:")
            for i in range(min(6, len(raw_df))):
                row_data = raw_df.iloc[i].tolist()
                # 只显示非空值
                non_empty = [(j, val) for j, val in enumerate(row_data) if val and str(val).strip()]
                print(f"    行{i}: {non_empty}")

            # 2. 寻找真正的表头行（现在是第2行，索引1）
            # 根据你的描述和输出，表头在第2行（索引1）
            header_row_idx = 1  # 第2行（0-based索引）

            # 验证这确实是表头行
            header_row = raw_df.iloc[header_row_idx].tolist()
            print(f"\n    第{header_row_idx + 1}行（候选表头）:")
            for i, header in enumerate(header_row):
                if header and str(header).strip():
                    print(f"      列{i}: '{header}'")

            # 检查是否包含关键表头
            header_str = ' '.join([str(h) for h in header_row if h])

            # 更灵活的表头检测逻辑
            header_keywords = ['展区', '作品名称', '设计作者', '序号']
            header_matches = sum([1 for keyword in header_keywords if keyword in header_str])

            if header_matches >= 2:  # 至少匹配2个关键词就认为是表头
                print(f"    ✅ 确认第{header_row_idx + 1}行是表头行")
            else:
                print(f"    ⚠️  第{header_row_idx + 1}行可能不是正确的表头行")
                # 尝试寻找包含关键词的行
                for idx in range(min(10, len(raw_df))):
                    row_str = ' '.join([str(v) for v in raw_df.iloc[idx].tolist() if v])
                    header_matches = sum([1 for keyword in header_keywords if keyword in row_str])
                    if header_matches >= 2:
                        header_row_idx = idx
                        header_row = raw_df.iloc[header_row_idx].tolist()
                        print(f"    🔍 找到新表头行: 第{idx + 1}行")
                        break

            # 3. 从表头下一行开始是数据
            data_start_row = header_row_idx + 1
            print(f"\n    数据起始行: {data_start_row}")
            print(f"    预计数据行数: {len(raw_df) - data_start_row}")

            # 4. 处理数据行
            current_item = {}

            for row_idx in range(data_start_row, len(raw_df)):
                row_data = raw_df.iloc[row_idx].tolist()

                # 跳过空行
                if not any(cell and str(cell).strip() for cell in row_data):
                    continue

                # 调试：显示当前行
                non_empty = [(i, val) for i, val in enumerate(row_data) if val and str(val).strip()]
                if non_empty:
                    print(f"\n    处理行{row_idx}（非空列）: {non_empty}")

                # 提取作品信息
                item_info = self._extract_item_info(row_data, header_row, sheet_name, row_idx)

                if item_info:
                    # 创建文档
                    item_docs = self._create_documents_for_item(item_info, sheet_name)
                    documents.extend(item_docs)

                    print(f"      为 '{item_info.get('作品名称', '未知')}' 创建了 {len(item_docs)} 个文档片段")

            print(f"\n    识别到 {len(documents)} 个文档片段")

        except Exception as e:
            print(f"    ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()

        return documents

    def _extract_item_info(self, row_data: list, header_row: list, sheet_name: str, row_idx: int = -1) -> Dict[str, Any]:
        """
        从行中提取作品信息

        Args:
            row_data: 行数据
            header_row: 表头行
            sheet_name: sheet名称
            row_idx: 行索引（用于调试）
        """
        item_info = {}

        # 根据表头映射字段
        for i, header in enumerate(header_row):
            if header and str(header).strip() and i < len(row_data):
                value = row_data[i]
                if value is not None and str(value).strip():
                    # 清理列名
                    col_name = str(header).strip().replace('\n', ' ').replace('\r', '')
                    item_info[col_name] = str(value).strip()

        # 如果没有提取到有效信息，返回空字典
        if not item_info:
            return {}

        # 添加sheet信息
        item_info['sheet_name'] = sheet_name
        item_info['category'] = self._map_sheet_to_category(sheet_name)

        # 特别处理：展区字段可能在第一列（列0）
        if '展区' not in item_info and len(row_data) > 0 and row_data[0]:
            item_info['展区'] = str(row_data[0]).strip()

        # 确保有作品名称
        if '作品名称' not in item_info or not item_info['作品名称']:
            print(f"      警告: 行{row_idx if row_idx >= 0 else '未知'}没有作品名称")
            return {}

        print(f"      提取作品: {item_info.get('作品名称', '未知')}")
        print(f"      展区: {item_info.get('展区', '未知')}")
        print(f"      作者: {item_info.get('设计作者', '未知')}")

        return item_info

    def _map_sheet_to_category(self, sheet_name: str) -> str:
        """
        映射sheet名称到类别
        """
        if sheet_name in self.category_map:
            return self.category_map[sheet_name]

        for key, value in self.category_map.items():
            if key in sheet_name:
                return value

        # 尝试从sheet名称中提取类别
        if '工业' in sheet_name:
            return '工业设计'
        elif '环境' in sheet_name:
            return '环境设计'
        elif '艺术' in sheet_name or '科技' in sheet_name:
            return '艺术与科技'

        return sheet_name.replace('类', '').strip()

    def _create_documents_for_item(self, item_info: Dict[str, Any], sheet_name: str) -> List[Document]:
        """
        为作品创建文档
        """
        documents = []

        item_name = item_info.get('作品名称', '').strip()
        if not item_name or item_name == '未知':
            print(f"      作品名称无效，跳过")
            return documents

        try:
            # 1. 基本信息文档
            basic_doc = self._create_basic_info_doc(item_info, sheet_name)
            if basic_doc:
                documents.append(basic_doc)

            # 2. 详细描述文档
            detailed_doc = self._create_detailed_info_doc(item_info, sheet_name)
            if detailed_doc:
                documents.append(detailed_doc)

            # 3. 设计理念文档
            concept_doc = self._create_design_concept_doc(item_info, sheet_name)
            if concept_doc:
                documents.append(concept_doc)

            # 4. 技术特点文档
            tech_doc = self._create_tech_info_doc(item_info, sheet_name)
            if tech_doc:
                documents.append(tech_doc)

            print(f"      为 '{item_name}' 创建了 {len(documents)} 个文档片段")

        except Exception as e:
            print(f"      创建文档失败: {e}")
            import traceback
            traceback.print_exc()

        return documents

    def _create_basic_info_doc(self, item_info: Dict[str, Any], sheet_name: str) -> Optional[Document]:
        """创建基本信息文档"""
        item_name = item_info.get('作品名称', '').strip()
        if not item_name:
            return None

        # 处理序号/点位字段
        item_id = item_info.get('序号/点位', '') or item_info.get('序号', '')

        # 构建内容
        content = f"""
【作品基本信息】

作品名称：{item_name}
展区位置：{item_info.get('展区', '')} - {item_id}
作品类别：{item_info.get('category', '')} / {item_info.get('类别标签', item_info.get('类别', ''))}
呈现形式：{item_info.get('呈现形式', '')}

设计作者：{item_info.get('设计作者', '')}
指导老师：{item_info.get('指导老师', '')}
创作时间：{item_info.get('创作时间', '')}

【作品简介】
{item_info.get('作品描述（简）', '暂无描述')}
"""

        # 构建元数据
        metadata = {
            "source": self.file_path,
            "sheet_name": sheet_name,
            "category": item_info.get('category', ''),
            "type": "basic_info",
            "item_name": item_name,
            "zone": item_info.get('展区', ''),
            "item_id": item_id,
            "sub_category": item_info.get('类别标签', item_info.get('类别', '')),
            "display_form": item_info.get('呈现形式', ''),
            "authors": item_info.get('设计作者', ''),
            "instructor": item_info.get('指导老师', ''),
            "creation_time": item_info.get('创作时间', '')
        }

        return Document(page_content=content.strip(), metadata=metadata)

    def _create_detailed_info_doc(self, item_info: Dict[str, Any], sheet_name: str) -> Optional[Document]:
        """创建详细描述文档"""
        # 收集所有详细字段
        detail_fields = [
            ("设计动机", item_info.get('设计动机', '')),
            ("灵感来源", item_info.get('灵感来源', '')),
            ("设计目的/意义", item_info.get('设计目的/意义', '')),
            ("创作历程", item_info.get('创作历程', '')),
            ("面临的困难", item_info.get('面临的困难', ''))
        ]

        # 过滤空字段
        valid_details = [(name, value) for name, value in detail_fields
                         if value and str(value).strip()]

        if not valid_details:
            return None

        # 处理序号/点位字段
        item_id = item_info.get('序号/点位', '') or item_info.get('序号', '')

        # 构建内容
        content = f"""
【作品详细描述】

作品名称：{item_info.get('作品名称', '')}
展区位置：{item_info.get('展区', '')} - {item_id}
作品类别：{item_info.get('category', '')}
"""

        for field_name, field_value in valid_details:
            content += f"\n【{field_name}】\n{field_value}\n"

        # 构建元数据
        metadata = {
            "source": self.file_path,
            "sheet_name": sheet_name,
            "category": item_info.get('category', ''),
            "type": "detailed_info",
            "item_name": item_info.get('作品名称', ''),
            "zone": item_info.get('展区', ''),
            "item_id": item_id,
            "has_details": True,
            "detail_fields": [name for name, _ in valid_details]
        }

        return Document(page_content=content.strip(), metadata=metadata)

    def _create_design_concept_doc(self, item_info: Dict[str, Any], sheet_name: str) -> Optional[Document]:
        """创建设计理念文档"""
        design_concept = item_info.get('设计理念/风格', '')
        visual_language = item_info.get('视觉形式语言', '')

        if not any([design_concept, visual_language]):
            return None

        # 构建内容
        content = f"""
【设计理念与视觉风格】

作品名称：{item_info.get('作品名称', '')}
作品类别：{item_info.get('category', '')}
"""

        if design_concept:
            content += f"\n设计理念：\n{design_concept}\n"

        if visual_language:
            content += f"\n视觉形式语言：\n{visual_language}\n"

        # 构建元数据
        metadata = {
            "source": self.file_path,
            "sheet_name": sheet_name,
            "category": item_info.get('category', ''),
            "type": "design_concept",
            "item_name": item_info.get('作品名称', ''),
            "has_design_concept": bool(design_concept),
            "has_visual_language": bool(visual_language)
        }

        return Document(page_content=content.strip(), metadata=metadata)

    def _create_tech_info_doc(self, item_info: Dict[str, Any], sheet_name: str) -> Optional[Document]:
        """创建技术特点文档"""
        technique = item_info.get('技术特点', '')
        expected_effect = item_info.get('预期效果', '')

        if not any([technique, expected_effect]):
            return None

        # 构建内容
        content = f"""
【技术特点与预期效果】

作品名称：{item_info.get('作品名称', '')}
作品类别：{item_info.get('category', '')}
"""

        if technique:
            content += f"\n技术特点：\n{technique}\n"

        if expected_effect:
            content += f"\n预期效果：\n{expected_effect}\n"

        # 构建元数据
        metadata = {
            "source": self.file_path,
            "sheet_name": sheet_name,
            "category": item_info.get('category', ''),
            "type": "tech_info",
            "item_name": item_info.get('作品名称', ''),
            "has_technique": bool(technique),
            "has_expected_effect": bool(expected_effect)
        }

        return Document(page_content=content.strip(), metadata=metadata)


def load_complex_exhibition_excel(data_dir: str = "data/raw_docs") -> List[Document]:
    """
    加载目录下的复杂格式展览Excel文件（新格式）
    """
    import glob

    all_documents = []

    # 查找Excel文件
    excel_files = (glob.glob(os.path.join(data_dir, "*.xlsx")) +
                   glob.glob(os.path.join(data_dir, "*.xls")) +
                   glob.glob(os.path.join(data_dir, "*.xlsm")))

    if not excel_files:
        print(f"⚠️  在 {data_dir} 中未找到Excel文件")
        return all_documents

    print(f"📂 发现 {len(excel_files)} 个Excel文件")

    for excel_file in excel_files:
        print(f"\n{'=' * 60}")

        try:
            loader = ComplexExhibitionExcelLoader(excel_file)
            documents = loader.load_all_sheets()
            all_documents.extend(documents)

            print(f"✅ {os.path.basename(excel_file)}: {len(documents)} 个文档")

        except Exception as e:
            print(f"❌ 处理失败 {excel_file}: {e}")
            import traceback
            traceback.print_exc()

    return all_documents


if __name__ == "__main__":
    # 模块测试
    print("🧪 complex_exhibition_excel_loader_v2.py 模块测试（新格式）")
    print("=" * 60)

    # 测试数据目录
    test_dir = "data/raw_docs"

    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        print(f"请创建目录并放入Excel文件")
    else:
        documents = load_complex_exhibition_excel(test_dir)

        if documents:
            print(f"\n✅ 测试成功")
            print(f"总文档数: {len(documents)}")

            # 统计文档类型
            doc_types = {}
            for doc in documents:
                doc_type = doc.metadata.get("type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            print(f"\n文档类型分布:")
            for doc_type, count in doc_types.items():
                print(f"  {doc_type}: {count}")

            # 显示示例
            print(f"\n示例文档:")
            for i, doc in enumerate(documents[:3]):
                print(f"\n[{i + 1}] {doc.metadata.get('item_name', '未知')}")
                print(f"类型: {doc.metadata.get('type', '未知')}")
                print(f"内容预览: {doc.page_content[:150]}...")
        else:
            print(f"❌ 未加载到文档")
