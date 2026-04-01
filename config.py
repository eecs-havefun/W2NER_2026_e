import json
from typing import List, Any, Dict, Optional


class Config:
    """安全的配置管理类，支持类型检查和动态属性验证"""
    
    # 定义所有已知配置字段及其类型（用于验证）
    KNOWN_FIELDS = {
        "dataset": str,
        "save_path": str,
        "predict_path": str,
        "dist_emb_size": int,
        "type_emb_size": int,
        "lstm_hid_size": int,
        "conv_hid_size": int,
        "bert_hid_size": int,
        "biaffine_size": int,
        "ffnn_hid_size": int,
        "dilation": (list, str),  # 可以是列表或字符串
        "emb_dropout": float,
        "conv_dropout": float,
        "out_dropout": float,
        "epochs": int,
        "batch_size": int,
        "learning_rate": float,
        "weight_decay": float,
        "clip_grad_norm": float,
        "bert_name": str,
        "bert_learning_rate": float,
        "warm_factor": float,
        "use_bert_last_4_layers": (bool, int),
        "seed": int,
    }
    
    # 新增字段（不在原始JSON中）
    EXTRA_FIELDS = {
        "data_root": str,  # 数据根目录
        "cache_dir": str,  # 缓存目录
        "continuous_only": bool,  # ProcNet相关配置
        "label_num": int,  # 运行时设置
        "vocab": object,  # 运行时设置
        "logger": Optional[Any],  # 运行时设置
    }
    
    def __init__(self, args):
        # 加载JSON配置
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 初始化所有已知字段为None
        for field_name in self.KNOWN_FIELDS:
            setattr(self, field_name, None)
        
        # 初始化额外字段
        self.data_root = "./data"  # 默认值
        self.cache_dir = "./cache"  # 缓存目录默认值
        self.continuous_only = True  # 默认值
        self.label_num = None
        self.vocab = None
        self.logger = None
        
        # 存储未知字段（保持向后兼容）
        self._extra_attrs = {}
        
        # 从JSON设置已知字段
        for key, value in config.items():
            self._safe_setattr(key, value)
        
        # 从命令行参数更新（覆盖JSON配置）
        for key, value in args.__dict__.items():
            if value is not None:
                self._safe_setattr(key, value)
        
        # 后处理：确保字段类型正确
        self._post_process()
    
    def _safe_setattr(self, key: str, value: Any) -> None:
        """安全地设置属性，支持类型检查和动态属性"""
        if key in self.KNOWN_FIELDS or key in self.EXTRA_FIELDS:
            # 已知字段，直接设置
            setattr(self, key, value)
        else:
            # 未知字段，存储到扩展属性
            self._extra_attrs[key] = value
    
    def _post_process(self) -> None:
        """后处理：确保字段类型正确"""
        # 处理 dilation 字段
        if isinstance(self.dilation, str):
            self.dilation = [int(x.strip()) for x in self.dilation.split(",")]
        elif not isinstance(self.dilation, list):
            self.dilation = [self.dilation]
        
        # 处理 use_bert_last_4_layers 字段
        if isinstance(self.use_bert_last_4_layers, int):
            self.use_bert_last_4_layers = bool(self.use_bert_last_4_layers)
    
    def __getattr__(self, name: str) -> Any:
        """支持向后兼容的动态属性访问"""
        if name in self._extra_attrs:
            return self._extra_attrs[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """支持设置动态属性"""
        # 允许设置任何属性，但如果是已知字段则进行类型检查
        if name in self.KNOWN_FIELDS and value is not None:
            expected_type = self.KNOWN_FIELDS[name]
            
            # 类型检查，允许数值类型转换
            if isinstance(value, expected_type):
                pass  # 类型匹配
            elif isinstance(expected_type, tuple) and any(isinstance(value, t) for t in expected_type):
                pass  # 匹配元组中的任一类型
            else:
                # 尝试数值类型转换
                if expected_type == float and isinstance(value, int):
                    value = float(value)
                elif expected_type == int and isinstance(value, float) and value.is_integer():
                    value = int(value)
                elif expected_type == bool and isinstance(value, int):
                    value = bool(value)
                else:
                    raise TypeError(f"Field '{name}' expects type {expected_type}, got {type(value)}")
        
        super().__setattr__(name, value)
    
    def get_data_path(self, dataset_type: str = "train") -> str:
        """获取数据文件路径"""
        import os
        return os.path.join(self.data_root, self.dataset, f"{dataset_type}.json")
    
    def validate(self) -> List[str]:
        """验证配置完整性，返回错误列表"""
        errors = []
        
        # 检查必需字段
        required_fields = ["dataset", "bert_name"]
        for field_name in required_fields:
            if getattr(self, field_name) is None:
                errors.append(f"Missing required field: {field_name}")
        
        # 检查数值范围
        if self.batch_size is not None and self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        
        if self.learning_rate is not None and self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        
        return errors
    
    def __repr__(self) -> str:
        """改进的字符串表示"""
        items = []
        
        # 添加已知字段
        for field_name in sorted(self.KNOWN_FIELDS.keys()):
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                items.append(f"{field_name}={value!r}")
        
        # 添加额外字段
        for field_name in sorted(self.EXTRA_FIELDS.keys()):
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                items.append(f"{field_name}={value!r}")
        
        # 添加扩展属性
        for key, value in sorted(self._extra_attrs.items()):
            items.append(f"{key}={value!r}")
        
        return f"Config({', '.join(items)})"
