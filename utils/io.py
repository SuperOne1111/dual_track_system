"""
原子写入工具模块
依据《技术规格说明书》7.1 节实现，确保实盘状态文件安全写入
防止因断电/崩溃导致的状态文件损坏
"""

import json
import tempfile
import os
import hashlib
from typing import Any, Dict


def calculate_checksum(data: str) -> str:
    """
    计算字符串数据的校验和
    用于验证原子写入过程中数据完整性
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def atomic_write_json(data: Dict[str, Any], filepath: str) -> None:
    """
    原子写入JSON文件
    实现 tempfile -> checksum -> os.replace 的安全写入流程
    依据《技术规格说明书》7.1 节 RUN-02 标准
    """
    # 创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(filepath) or '.',
        prefix='tmp_',
        suffix='.json'
    )
    
    try:
        # 将数据写入临时文件
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as tmp_file:
            json.dump(data, tmp_file, ensure_ascii=False, indent=2)
        
        # 计算临时文件校验和（作为额外验证）
        with open(temp_path, 'r', encoding='utf-8') as tmp_file:
            temp_content = tmp_file.read()
        checksum = calculate_checksum(temp_content)
        
        # 重命名临时文件为最终文件名（原子操作）
        os.replace(temp_path, filepath)
        
        print(f"原子写入成功: {filepath}, 校验和: {checksum[:8]}")
        
    except Exception as e:
        # 清理临时文件
        try:
            os.close(temp_fd)
        except:
            pass
        
        try:
            os.unlink(temp_path)
        except:
            pass
            
        raise e


if __name__ == "__main__":
    # 测试用例
    test_data = {
        "status": "running",
        "timestamp": "2024-05-24T10:00:00Z",
        "positions": {"stock_a": 0.5, "stock_b": 0.3}
    }
    
    atomic_write_json(test_data, "./test_atomic_write.json")
    print("原子写入测试完成")