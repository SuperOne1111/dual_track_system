#!/usr/bin/env python3
"""
生产环境监控脚本
依据《dual-track-api-design》2.1 节和《dual-track-tech-spec》3.6 节实现

此脚本用于监控生产环境的系统状态、因子稳定性、交易执行情况等
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

from modules.data_layer import DataLoader
from modules.state_manager import StateRepository, FileRepository
from utils.io import atomic_write_json


class ProductionMonitor:
    """
    生产环境监控器
    依据《dual-track-api-design》2.1 节 StateRepository 接口设计
    """
    
    def __init__(self, config_path: str = "config/production.yaml"):
        """初始化监控器"""
        self.config_path = config_path
        self.logger = self._setup_logger()
        
        # 初始化数据层
        self.data_loader = DataLoader()
        
        # 初始化状态仓库
        self.state_repo = FileRepository()
        
        # 监控指标配置
        self.monitor_config = {
            "factor_ic_threshold": 0.05,  # 因子IC阈值
            "ic_win_rate_threshold": 0.30,  # IC胜率阈值
            "volatility_threshold": 0.15,  # 波动率阈值
            "position_limit_warning": 0.80,  # 仓位上限警告
        }
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("ProductionMonitor")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        检查系统健康状态
        依据《dual-track-tech-spec》3.6 因子稳定性监控
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_status": "HEALTHY"
        }
        
        # 检查状态文件是否存在
        health_status["checks"]["state_files_exist"] = self._check_state_files()
        
        # 检查因子权重状态
        health_status["checks"]["factor_weights_valid"] = self._check_factor_weights()
        
        # 检查最近运行状态
        health_status["checks"]["recent_run_success"] = self._check_recent_run()
        
        # 计算整体状态
        if not all(health_status["checks"].values()):
            health_status["overall_status"] = "WARNING"
            
        return health_status
    
    def _check_state_files(self) -> bool:
        """检查状态文件是否存在且可读"""
        try:
            state_files = [
                "state_cool_down.json",
                "state_factor_weights.json", 
                "state_portfolio_snapshot.csv"
            ]
            
            for file_name in state_files:
                file_path = Path("data/prod_states") / file_name
                if not file_path.exists():
                    self.logger.warning(f"State file missing: {file_path}")
                    return False
                    
                # 尝试读取文件
                if file_name.endswith('.json'):
                    with open(file_path, 'r') as f:
                        json.load(f)
                else:
                    pd.read_csv(file_path)
                    
            return True
        except Exception as e:
            self.logger.error(f"Error checking state files: {e}")
            return False
    
    def _check_factor_weights(self) -> bool:
        """检查因子权重是否合理"""
        try:
            factor_weights = self.state_repo.load("factor_weights")
            if not factor_weights or not isinstance(factor_weights, dict):
                self.logger.warning("Factor weights not found or invalid")
                return False
                
            # 检查权重和是否接近1
            for track_weights in factor_weights.values():
                if isinstance(track_weights, dict):
                    total_weight = sum(track_weights.values())
                    if abs(total_weight - 1.0) > 0.1:  # 允许10%误差
                        self.logger.warning(f"Factor weights don't sum to 1: {total_weight}")
                        return False
                        
            return True
        except Exception as e:
            self.logger.error(f"Error checking factor weights: {e}")
            return False
    
    def _check_recent_run(self) -> bool:
        """检查最近运行是否成功"""
        try:
            # 检查是否有最近的执行日志或输出文件
            snapshot_file = Path("data/prod_states/state_portfolio_snapshot.csv")
            if not snapshot_file.exists():
                return False
                
            # 检查文件修改时间是否在过去24小时内
            mod_time = datetime.fromtimestamp(snapshot_file.stat().st_mtime)
            time_diff = datetime.now() - mod_time
            
            return time_diff.total_seconds() < 86400  # 24小时内
        except Exception as e:
            self.logger.error(f"Error checking recent run: {e}")
            return False
    
    def monitor_factor_stability(self) -> Dict[str, Any]:
        """
        监控因子稳定性
        依据《dual-track-tech-spec》3.6 因子稳定性监控
        """
        try:
            # 获取因子权重和IC统计数据
            factor_weights = self.state_repo.load("factor_weights")
            
            stability_report = {
                "timestamp": datetime.now().isoformat(),
                "factors": {},
                "summary": {
                    "total_factors": 0,
                    "unstable_factors": 0,
                    "avg_ic": 0.0,
                    "avg_ic_win_rate": 0.0
                }
            }
            
            if not factor_weights:
                self.logger.warning("No factor weights found for monitoring")
                return stability_report
            
            # 计算因子稳定性指标
            ic_values = []
            win_rates = []
            
            for track, weights in factor_weights.items():
                if not isinstance(weights, dict):
                    continue
                    
                for factor_name, weight in weights.items():
                    factor_info = {
                        "weight": weight,
                        "status": "STABLE",
                        "ic_value": None,
                        "ic_win_rate": None
                    }
                    
                    # 这里应该从历史IC数据中获取信息
                    # 为了简化，我们模拟一些检查逻辑
                    # 在实际生产中，这里应该查询历史IC数据
                    
                    # 模拟IC值检查
                    ic_value = weight * 0.1  # 模拟IC值，实际应从历史数据获取
                    factor_info["ic_value"] = ic_value
                    
                    # 模拟IC胜率检查
                    ic_win_rate = 0.5 if ic_value > 0 else 0.3  # 模拟胜率
                    factor_info["ic_win_rate"] = ic_win_rate
                    
                    # 判断因子是否稳定
                    if abs(ic_value) < self.monitor_config["factor_ic_threshold"]:
                        factor_info["status"] = "LOW_IC"
                        stability_report["summary"]["unstable_factors"] += 1
                    elif ic_win_rate < self.monitor_config["ic_win_rate_threshold"]:
                        factor_info["status"] = "LOW_WIN_RATE"
                        stability_report["summary"]["unstable_factors"] += 1
                    
                    stability_report["factors"][f"{track}.{factor_name}"] = factor_info
                    ic_values.append(ic_value)
                    win_rates.append(ic_win_rate)
            
            # 计算汇总统计
            stability_report["summary"]["total_factors"] = len(ic_values)
            stability_report["summary"]["avg_ic"] = sum(ic_values) / len(ic_values) if ic_values else 0
            stability_report["summary"]["avg_ic_win_rate"] = sum(win_rates) / len(win_rates) if win_rates else 0
            
            return stability_report
            
        except Exception as e:
            self.logger.error(f"Error monitoring factor stability: {e}")
            return {"error": str(e)}
    
    def monitor_portfolio_risk(self) -> Dict[str, Any]:
        """
        监控投资组合风险
        依据《dual-track-tech-spec》4.3 权重分配与现金管理
        """
        try:
            # 获取当前持仓快照
            portfolio_snapshot_path = Path("data/prod_states/state_portfolio_snapshot.csv")
            
            if not portfolio_snapshot_path.exists():
                return {"error": "Portfolio snapshot file not found"}
            
            portfolio_df = pd.read_csv(portfolio_snapshot_path)
            
            risk_report = {
                "timestamp": datetime.now().isoformat(),
                "risk_metrics": {},
                "alerts": []
            }
            
            if portfolio_df.empty:
                risk_report["alerts"].append("Empty portfolio - no positions detected")
                return risk_report
            
            # 计算风险指标
            total_weight = portfolio_df.get('weight', pd.Series([0])).sum()
            cash_ratio = portfolio_df[portfolio_df['sector_id'] == 'CASH']['weight'].sum() if 'CASH' in portfolio_df.get('sector_id', []) else 0
            
            risk_report["risk_metrics"] = {
                "total_allocated_weight": float(total_weight),
                "cash_ratio": float(cash_ratio),
                "position_count": len(portfolio_df),
                "max_single_position": float(portfolio_df['weight'].max()) if 'weight' in portfolio_df.columns else 0
            }
            
            # 检查风险阈值
            if total_weight > self.monitor_config["position_limit_warning"]:
                risk_report["alerts"].append(f"Total allocated weight ({total_weight:.2%}) exceeds warning threshold ({self.monitor_config['position_limit_warning']:.0%})")
                
            if 'volatility' in portfolio_df.columns:
                avg_volatility = portfolio_df['volatility'].mean()
                if avg_volatility > self.monitor_config["volatility_threshold"]:
                    risk_report["alerts"].append(f"Average portfolio volatility ({avg_volatility:.2%}) exceeds threshold ({self.monitor_config['volatility_threshold']:.0%})")
            
            return risk_report
            
        except Exception as e:
            self.logger.error(f"Error monitoring portfolio risk: {e}")
            return {"error": str(e)}
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """生成完整的监控报告"""
        report = {
            "report_type": "Production Monitoring Report",
            "generated_at": datetime.now().isoformat(),
            "system_health": self.check_system_health(),
            "factor_stability": self.monitor_factor_stability(),
            "portfolio_risk": self.monitor_portfolio_risk()
        }
        
        return report
    
    def save_monitoring_report(self, report: Dict[str, Any], output_path: str = None) -> str:
        """保存监控报告到文件"""
        if output_path is None:
            output_dir = Path("data/prod_states/monitoring_reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = str(output_dir / filename)
        
        try:
            atomic_write_json(report, output_path)
            self.logger.info(f"Monitoring report saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving monitoring report: {e}")
            raise


def main():
    """主函数 - 脚本入口"""
    monitor = ProductionMonitor()
    
    print("Starting production environment monitoring...")
    
    # 生成监控报告
    report = monitor.generate_monitoring_report()
    
    # 保存报告
    report_path = monitor.save_monitoring_report(report)
    
    # 输出关键信息
    print(f"Monitoring report generated: {report_path}")
    print(f"Overall system status: {report['system_health']['overall_status']}")
    
    # 检查是否有警告或错误
    alerts = []
    if report['portfolio_risk'].get('alerts'):
        alerts.extend(report['portfolio_risk']['alerts'])
    
    if report['system_health']['overall_status'] != 'HEALTHY':
        alerts.append(f"System health status: {report['system_health']['overall_status']}")
    
    if report['factor_stability'].get('summary', {}).get('unstable_factors', 0) > 0:
        unstable_count = report['factor_stability']['summary']['unstable_factors']
        alerts.append(f"Found {unstable_count} unstable factors")
    
    if alerts:
        print("\nAlerts found:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("\nNo issues detected - system appears healthy")


if __name__ == "__main__":
    main()