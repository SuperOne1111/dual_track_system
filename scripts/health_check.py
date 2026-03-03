#!/usr/bin/env python3
"""
系统健康检查脚本
依据《dual-track-api-design》2.1 节和《dual-track-tech-spec》相关章节实现

此脚本用于执行快速的系统健康检查，验证基本功能是否正常
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

from modules.data_layer import DataLoader
from modules.state_manager import FileRepository
from modules.buy_signal_generator import TrendScreener, LeftScreener
from modules.sell_signal_generator import RiskManager
from modules.risk_manager import PortfolioBuilder
from utils.io import atomic_write_json


class HealthChecker:
    """
    系统健康检查器
    依据《dual-track-api-design》2.1 节相关模块接口设计
    """
    
    def __init__(self):
        """初始化健康检查器"""
        self.logger = self._setup_logger()
        self.data_loader = DataLoader()
        self.state_repo = FileRepository()
        
        # 检查配置
        self.check_config = {
            "min_data_days": 60,  # 最少数据天数
            "max_check_time": 30,  # 最大检查时间（秒）
            "required_files": [
                "data/prod_states/state_cool_down.json",
                "data/prod_states/state_factor_weights.json",
            ]
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("HealthChecker")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """运行全面健康检查"""
        start_time = datetime.now()
        
        health_report = {
            "report_type": "System Health Check Report",
            "generated_at": start_time.isoformat(),
            "checks": {},
            "overall_status": "HEALTHY",
            "execution_time": 0
        }
        
        # 执行各项检查
        health_report["checks"]["environment"] = self.check_environment()
        health_report["checks"]["dependencies"] = self.check_dependencies()
        health_report["checks"]["data_access"] = self.check_data_access()
        health_report["checks"]["state_management"] = self.check_state_management()
        health_report["checks"]["core_logic"] = self.check_core_logic()
        
        # 计算整体状态
        all_checks_passed = all(check.get("status") == "PASS" for check in health_report["checks"].values())
        health_report["overall_status"] = "HEALTHY" if all_checks_passed else "UNHEALTHY"
        
        # 计算执行时间
        end_time = datetime.now()
        health_report["execution_time"] = (end_time - start_time).total_seconds()
        
        return health_report
    
    def check_environment(self) -> Dict[str, Any]:
        """检查运行环境"""
        check_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # 检查Python版本
            import sys
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
                check_result["status"] = "FAIL"
                check_result["errors"].append(f"Python version {python_version} is too old, requires 3.8+")
            else:
                check_result["details"]["python_version"] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            # 检查必要目录
            required_dirs = ["data", "data/prod_states", "config", "modules", "utils"]
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    check_result["status"] = "WARN"
                    check_result["warnings"].append(f"Directory {dir_name} does not exist")
                else:
                    check_result["details"][f"dir_{dir_name}"] = "exists"
            
            # 检查必要文件
            for file_path in self.check_config["required_files"]:
                if not Path(file_path).exists():
                    check_result["status"] = "WARN" 
                    check_result["warnings"].append(f"Required file {file_path} does not exist")
                else:
                    check_result["details"][f"file_{Path(file_path).name}"] = "exists"
                    
        except Exception as e:
            check_result["status"] = "FAIL"
            check_result["errors"].append(f"Environment check failed: {str(e)}")
        
        return check_result
    
    def check_dependencies(self) -> Dict[str, Any]:
        """检查依赖包"""
        check_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        required_packages = [
            ("pandas", "pd"),
            ("numpy", "np"),
            ("psycopg2", "psycopg2"),  # 数据库连接
        ]
        
        for package_name, import_name in required_packages:
            try:
                if package_name == "psycopg2":
                    import psycopg2
                    globals()[import_name] = psycopg2
                elif package_name == "pandas":
                    import pandas as pd
                    globals()[import_name] = pd
                elif package_name == "numpy":
                    import numpy as np
                    globals()[import_name] = np
                check_result["details"][package_name] = "available"
            except ImportError:
                check_result["status"] = "FAIL"
                check_result["errors"].append(f"Package {package_name} not available")
        
        return check_result
    
    def check_data_access(self) -> Dict[str, Any]:
        """检查数据访问能力"""
        check_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # 检查能否连接数据库并获取少量数据
            # 使用最近的日期获取数据以测试连接
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            # 尝试加载数据
            market_data = self.data_loader.load_market_data(
                start_date=start_date,
                end_date=end_date,
                as_of_date=end_date
            )
            
            if market_data is None or len(market_data) == 0:
                check_result["status"] = "WARN"
                check_result["warnings"].append("Could not load market data, connection might be unavailable")
            else:
                check_result["details"]["market_data_rows"] = len(market_data)
                check_result["details"]["data_date_range"] = {
                    "start": str(market_data['trade_date'].min()),
                    "end": str(market_data['trade_date'].max())
                }
                
            # 检查能否获取行业元数据
            sector_meta = self.data_loader.load_sector_metadata()
            if sector_meta is None or len(sector_meta) == 0:
                check_result["status"] = "WARN"
                check_result["warnings"].append("Could not load sector metadata")
            else:
                check_result["details"]["sector_count"] = len(sector_meta)
                
        except Exception as e:
            check_result["status"] = "FAIL"
            check_result["errors"].append(f"Data access check failed: {str(e)}")
        
        return check_result
    
    def check_state_management(self) -> Dict[str, Any]:
        """检查状态管理功能"""
        check_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # 测试状态读取
            test_key = "factor_weights"
            state_data = self.state_repo.load(test_key)
            
            if state_data is not None:
                check_result["details"]["loaded_state_keys"] = list(state_data.keys()) if isinstance(state_data, dict) else "loaded"
            else:
                check_result["warnings"].append(f"No state data found for key: {test_key}")
            
            # 测试状态写入（使用临时测试数据）
            test_data = {
                "test_timestamp": datetime.now().isoformat(),
                "test_value": "health_check"
            }
            
            # 使用临时文件测试写入功能
            temp_file = Path("data/prod_states/test_health_write.json")
            atomic_write_json(test_data, str(temp_file))
            
            # 验证写入是否成功
            if temp_file.exists():
                with open(temp_file, 'r') as f:
                    written_data = json.load(f)
                    if written_data.get("test_value") == "health_check":
                        check_result["details"]["write_test"] = "successful"
                    else:
                        check_result["status"] = "FAIL"
                        check_result["errors"].append("State write test failed - data not written correctly")
                
                # 清理临时文件
                temp_file.unlink()
            else:
                check_result["status"] = "FAIL"
                check_result["errors"].append("State write test failed - file not created")
                
        except Exception as e:
            check_result["status"] = "FAIL"
            check_result["errors"].append(f"State management check failed: {str(e)}")
        
        return check_result
    
    def check_core_logic(self) -> Dict[str, Any]:
        """检查核心逻辑功能"""
        check_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # 测试筛选器实例化
            trend_screener = TrendScreener()
            left_screener = LeftScreener()
            
            check_result["details"]["screener_init"] = "successful"
            
            # 测试风险管理器实例化
            risk_manager = RiskManager()
            check_result["details"]["risk_manager_init"] = "successful"
            
            # 测试组合构建器实例化
            portfolio_builder = PortfolioBuilder()
            check_result["details"]["portfolio_builder_init"] = "successful"
            
            # 简单的功能测试（不执行实际计算，仅验证接口可用性）
            if hasattr(trend_screener, 'screen_candidates'):
                check_result["details"]["trend_screener_interface"] = "available"
            else:
                check_result["warnings"].append("TrendScreener.screen_candidates method not found")
                
            if hasattr(left_screener, 'build_candidate_pool'):
                check_result["details"]["left_screener_interface"] = "available"
            else:
                check_result["warnings"].append("LeftScreener.build_candidate_pool method not found")
                
            if hasattr(risk_manager, 'should_sell_atr_drawdown'):
                check_result["details"]["risk_manager_interface"] = "available"
            else:
                check_result["warnings"].append("RiskManager.should_sell_atr_drawdown method not found")
                
        except Exception as e:
            check_result["status"] = "FAIL"
            check_result["errors"].append(f"Core logic check failed: {str(e)}")
        
        return check_result
    
    def save_health_report(self, report: Dict[str, Any], output_path: str = None) -> str:
        """保存健康检查报告到文件"""
        if output_path is None:
            output_dir = Path("data/prod_states/health_checks")
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"health_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = str(output_dir / filename)
        
        try:
            atomic_write_json(report, output_path)
            self.logger.info(f"Health check report saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving health check report: {e}")
            raise


def main():
    """主函数 - 脚本入口"""
    checker = HealthChecker()
    
    print("Starting system health check...")
    
    # 执行健康检查
    report = checker.run_comprehensive_check()
    
    # 保存报告
    report_path = checker.save_health_report(report)
    
    # 输出检查结果摘要
    print(f"Health check report generated: {report_path}")
    print(f"Overall status: {report['overall_status']}")
    print(f"Execution time: {report['execution_time']:.2f}s")
    
    # 输出详细结果
    print("\nDetailed results:")
    for check_name, check_result in report['checks'].items():
        status = check_result['status']
        print(f"  {check_name}: {status}")
        
        if check_result.get('errors'):
            for error in check_result['errors']:
                print(f"    ERROR: {error}")
        
        if check_result.get('warnings'):
            for warning in check_result['warnings']:
                print(f"    WARNING: {warning}")
    
    # 根据整体状态返回适当的退出码
    if report['overall_status'] == 'HEALTHY':
        print("\nSystem health check PASSED")
        return 0
    else:
        print("\nSystem health check FAILED or has WARNINGS")
        return 1 if report['overall_status'] == 'FAIL' else 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)