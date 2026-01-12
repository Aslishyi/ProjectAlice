from typing import Dict, List, Optional, Type, Any
from app.plugins.base_plugin import BasePlugin, PluginInfo
import logging
import os
import importlib.util

logger = logging.getLogger("PluginManager")


class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        """初始化插件管理器"""
        self._plugins: Dict[str, BasePlugin] = {}  # 插件名称 -> 插件实例
        self._plugin_classes: Dict[str, Type[BasePlugin]] = {}  # 插件名称 -> 插件类
        
        logger.info("插件管理器初始化完成")
    
    def load_plugins_from_directory(self, directory: str) -> int:
        """从目录加载插件
        
        Args:
            directory: 插件目录
            
        Returns:
            int: 加载的插件数量
        """
        if not os.path.exists(directory):
            logger.warning(f"插件目录 '{directory}' 不存在")
            return 0
        
        loaded_count = 0
        
        # 遍历目录下的所有子目录和.py文件
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            if os.path.isdir(item_path):
                # 尝试加载插件目录
                if self._load_plugin_directory(item_path):
                    loaded_count += 1
            elif os.path.isfile(item_path) and item.endswith(".py") and not item.startswith("_"):
                # 尝试加载单个插件文件
                if self._load_plugin_file(item_path):
                    loaded_count += 1
        
        logger.info(f"从目录 '{directory}' 加载了 {loaded_count} 个插件")
        return loaded_count
    
    def _load_plugin_file(self, file_path: str) -> bool:
        """加载单个插件文件
        
        Args:
            file_path: 插件文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # 加载模块
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                logger.error(f"无法加载插件文件 '{file_path}'")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找BasePlugin的子类
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj != BasePlugin:
                    # 检查插件是否已存在
                    if obj.plugin_info.name in self._plugin_classes:
                        logger.warning(f"插件 '{obj.plugin_info.name}' 已存在，跳过加载")
                        continue
                    
                    self._plugin_classes[obj.plugin_info.name] = obj
                    logger.info(f"加载插件类: {obj.plugin_info.name} v{obj.plugin_info.version}")
                    return True
            
            logger.warning(f"插件文件 '{file_path}' 中未找到 BasePlugin 的子类")
            return False
        except Exception as e:
            logger.error(f"加载插件文件 '{file_path}' 失败: {e}")
            return False
    
    def _load_plugin_directory(self, directory_path: str) -> bool:
        """加载插件目录
        
        Args:
            directory_path: 插件目录路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 检查是否有 __init__.py 文件
            init_file = os.path.join(directory_path, "__init__.py")
            if not os.path.exists(init_file):
                logger.warning(f"插件目录 '{directory_path}' 缺少 __init__.py 文件")
                return False
            
            # 加载模块
            # 使用完整的模块路径作为模块名称
            plugin_dir = os.path.basename(directory_path)
            module_name = f"app.plugins.{plugin_dir}"
            spec = importlib.util.spec_from_file_location(module_name, init_file)
            if not spec or not spec.loader:
                logger.error(f"无法加载插件目录 '{directory_path}'")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找BasePlugin的子类
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj != BasePlugin:
                    # 检查插件是否已存在
                    if obj.plugin_info.name in self._plugin_classes:
                        logger.warning(f"插件 '{obj.plugin_info.name}' 已存在，跳过加载")
                        continue
                    
                    self._plugin_classes[obj.plugin_info.name] = obj
                    logger.info(f"加载插件类: {obj.plugin_info.name} v{obj.plugin_info.version}")
                    return True
            
            logger.warning(f"插件目录 '{directory_path}' 中未找到 BasePlugin 的子类")
            return False
        except Exception as e:
            logger.error(f"加载插件目录 '{directory_path}' 失败: {e}")
            return False
    
    async def initialize_plugins(self) -> int:
        """初始化所有插件
        
        Returns:
            int: 初始化成功的插件数量
        """
        initialized_count = 0
        
        for plugin_name, plugin_class in self._plugin_classes.items():
            try:
                # 创建插件实例
                plugin = plugin_class()
                
                # 初始化插件
                if await plugin.initialize():
                    self._plugins[plugin_name] = plugin
                    initialized_count += 1
                else:
                    logger.error(f"插件 '{plugin_name}' 初始化失败")
            except Exception as e:
                logger.error(f"初始化插件 '{plugin_name}' 时发生错误: {e}")
        
        logger.info(f"初始化了 {initialized_count}/{len(self._plugin_classes)} 个插件")
        return initialized_count
    
    async def shutdown_plugins(self) -> int:
        """关闭所有插件
        
        Returns:
            int: 关闭成功的插件数量
        """
        shutdown_count = 0
        
        for plugin_name, plugin in list(self._plugins.items()):
            try:
                if await plugin.shutdown():
                    del self._plugins[plugin_name]
                    shutdown_count += 1
                else:
                    logger.error(f"插件 '{plugin_name}' 关闭失败")
            except Exception as e:
                logger.error(f"关闭插件 '{plugin_name}' 时发生错误: {e}")
        
        logger.info(f"关闭了 {shutdown_count}/{len(self._plugins)} 个插件")
        return shutdown_count
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """获取插件实例
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Optional[BasePlugin]: 插件实例或None
        """
        return self._plugins.get(plugin_name)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """获取插件信息
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Optional[PluginInfo]: 插件信息或None
        """
        plugin = self.get_plugin(plugin_name)
        return plugin.get_plugin_info() if plugin else None
    
    def get_all_plugins(self) -> Dict[str, BasePlugin]:
        """获取所有插件实例
        
        Returns:
            Dict[str, BasePlugin]: 插件名称 -> 插件实例
        """
        return self._plugins.copy()
    
    def get_all_plugin_classes(self) -> Dict[str, Type[BasePlugin]]:
        """获取所有插件类
        
        Returns:
            Dict[str, Type[BasePlugin]]: 插件名称 -> 插件类
        """
        return self._plugin_classes.copy()
    
    def get_plugins_stats(self) -> Dict[str, Any]:
        """获取插件统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "total_plugins": len(self._plugin_classes),
            "loaded_plugins": len(self._plugins),
            "plugin_list": [
                {
                    "name": plugin.plugin_info.name,
                    "version": plugin.plugin_info.version,
                    "description": plugin.plugin_info.description,
                    "author": plugin.plugin_info.author,
                    "enabled": plugin.plugin_info.enabled,
                    "tools": plugin.plugin_info.tools
                }
                for plugin in self._plugins.values()
            ]
        }


# 创建全局插件管理器实例
plugin_manager = PluginManager()