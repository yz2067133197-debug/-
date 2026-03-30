# bluetooth_data_collector.py
import asyncio
import csv
import threading
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Callable
import sys
import os
import socket
import subprocess
import re

# 导入messagebox模块
import tkinter.messagebox as messagebox

BLE_AVAILABLE = False
try:
    from bleak import BleakScanner, BleakClient
    BLE_AVAILABLE = True
except ImportError:
    pass  # 静默失败

class BaseBluetoothDataCollector:
    """蓝牙数据采集器基类 - 定义通用接口"""
    
    def __init__(self, log_callback: Optional[Callable] = None, 
                 data_callback: Optional[Callable] = None):
        """初始化蓝牙数据采集器"""
        self.log_callback = log_callback or print
        self.data_callback = data_callback
        
        # 蓝牙连接相关
        self.target_device_name = None
        self.target_device_address = None
        self.connected = False
        
        # 数据采集相关
        self.collected_data = []  # 存储采集的数据
        self.file_path = 'a.csv'  # 保存文件路径
        self.running = False
        self.collection_thread = None
        self.target_points = 0
        
        # 回调函数
        self.progress_callback = None
        self.completion_callback = None
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
        self.log_callback("蓝牙数据采集器初始化完成")
    
    def get_current_unit(self, unit_code: str) -> float:
        """将单位代码转换为实际的单位系数（参考Save.py的实现）"""
        units = {
            '1': 1e-12,  # pA
            '2': 1e-9,   # nA
            '3': 1e-6,   # μA
            '4': 1e-3    # mA
        }
        return units.get(unit_code, 1.0)  # 默认返回1.0，如果单位代码未知
    
    def parse_data_line(self, line: str) -> Optional[Dict]:
        """解析接收到的数据行，支持多通道（八通道）模式，参考Save.py的实现"""
        try:
            line = line.strip()
            if not line:
                return None
            
            # 只处理以VAL开头的数据行
            if not line.startswith('VAL'):
                return None
            
            # 分割数据部分
            parts = line.split(',')[1:]  # 跳过'VAL'部分
            
            # 多通道模式处理（参考Save.py的实现）
            if len(parts) > 4:  # 多通道模式：[电流值, 单位代码, 电流值, 单位代码, ...]
                # 处理每一对电流值和单位代码
                current_values = []
                for i in range(0, len(parts), 2):
                    if i + 1 < len(parts):  # 确保有对应的单位代码
                        try:
                            current_value = float(parts[i].strip())
                            unit_code = parts[i + 1].strip()
                            unit_coeff = self.get_current_unit(unit_code)
                            
                            # 转换为实际电流值
                            actual_current = current_value * unit_coeff
                            current_values.append(actual_current)
                        except ValueError:
                            # 如果解析单个通道失败，跳过该通道
                            continue
                
                if current_values:  # 确保至少有一个有效的通道数据
                    # 创建数据点，多通道模式
                    data_point = {
                        'Time': time.time() - getattr(self, 'start_time', 0),  # 使用相对时间
                        'Data': current_values,  # 多通道数据
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # 添加每个通道的详细信息
                    for i, current in enumerate(current_values):
                        data_point[f'channel_{i + 1}'] = current
                    
                    return data_point
            
            # 单通道模式处理（兼容旧格式）
            if len(parts) >= 3:  # 至少需要通道号、电压值和单位代码
                try:
                    # 获取通道号和电压值
                    channel = parts[0].strip()
                    voltage = float(parts[1].strip())
                    unit_code = parts[2].strip()
                    
                    # 获取单位系数并计算实际电流值
                    current = voltage
                    unit_coeff = self.get_current_unit(unit_code)
                    
                    # 转换为实际电流值（根据单位系数）
                    actual_current = current * unit_coeff
                    
                    # 创建数据点，单通道模式
                    data_point = {
                        'Time': time.time() - getattr(self, 'start_time', 0),  # 使用相对时间
                        'Data': [actual_current],  # 单通道数据
                        'timestamp': datetime.now().isoformat(),
                        'channel_1': actual_current  # 为了兼容，添加通道1的字段
                    }
                    
                    # 可选字段，仅在有值时添加
                    try:
                        data_point['Voltage'] = voltage
                        data_point['UnitCode'] = unit_code
                        data_point['Channel'] = channel
                    except Exception:
                        pass  # 忽略字段添加失败的情况
                    
                    return data_point
                except (ValueError, IndexError):
                    # 如果解析失败，返回None
                    pass
            
            return None
            
        except Exception as e:
            self.logger.debug(f"数据解析失败: {line} - {str(e)}")
            return None
    
    def save_to_csv(self, data: List[Dict], filename: str = None) -> bool:
        """将数据保存到CSV文件，优化多通道数据的保存"""
        if not filename:
            filename = self.file_path
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                if not data:
                    self.log_callback("没有数据需要保存")
                    return False
                
                # 确定CSV的列名
                # 首先检查是否有多通道字段
                fieldnames = ['Time']
                channel_fields = []
                
                # 找出所有channel_*字段
                for point in data:
                    for key in point.keys():
                        if key.startswith('channel_') and key not in channel_fields:
                            channel_fields.append(key)
                
                # 按通道号排序
                channel_fields.sort(key=lambda x: int(x.split('_')[1]))
                
                # 如果没有channel_字段，则使用Data列表中的通道
                if not channel_fields:
                    max_channels = max(len(point['Data']) for point in data)
                    channel_fields = [f'Current_{i+1}' for i in range(max_channels)]
                
                # 组合完整的字段名
                fieldnames.extend(channel_fields)
                
                # 添加额外信息字段（如果存在）
                extra_fields = ['Voltage', 'UnitCode', 'Channel']
                for field in extra_fields:
                    if field in data[0]:
                        fieldnames.append(field)
                
                # 创建CSV写入器
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 写入标题
                writer.writeheader()
                
                # 写入数据
                for point in data:
                    # 优化时间格式，保留3位小数
                    time_value = point['Time']
                    if isinstance(time_value, (int, float)):
                        # 保留3位小数，足够精确且显示更简洁
                        formatted_time = f"{time_value:.3f}"
                    else:
                        formatted_time = time_value
                    row = {'Time': formatted_time}
                    
                    # 优先使用channel_*字段
                    for channel_field in channel_fields:
                        if channel_field in point:
                            # 对于浮点数类型的数据，使用合适的显示格式
                            value = point[channel_field]
                            if isinstance(value, float):
                                # 根据值的大小选择合适的格式，避免过长的科学计数法
                                if abs(value) >= 1e-3 or abs(value) == 0:
                                    # 较大的值或0，使用普通格式
                                    row[channel_field] = f"{value:.10f}"
                                else:
                                    # 较小的值，使用简短的科学计数法（保留5位有效数字）
                                    row[channel_field] = f"{value:.5g}"
                            else:
                                row[channel_field] = value
                        # 如果没有channel_字段但有Data列表
                        elif 'Data' in point and channel_field.startswith('Current_'):
                            try:
                                channel_index = int(channel_field.split('_')[1]) - 1
                                if channel_index < len(point['Data']):
                                    value = point['Data'][channel_index]
                                    if isinstance(value, float):
                                        # 同样对Data列表中的浮点数进行格式控制
                                        if abs(value) >= 1e-3 or abs(value) == 0:
                                            row[channel_field] = f"{value:.10f}"
                                        else:
                                            row[channel_field] = f"{value:.5g}"
                                    else:
                                        row[channel_field] = value
                            except (ValueError, IndexError):
                                pass
                    
                    # 添加额外信息
                    for field in extra_fields:
                        if field in point:
                            row[field] = point[field]
                    
                    writer.writerow(row)
                
            self.log_callback(f"数据已保存到 {filename} ({len(data)} 条记录)")
            return True
            
        except Exception as e:
            self.log_callback(f"保存数据失败: {str(e)}")
            return False
    
    def is_running(self) -> bool:
        """检查是否正在采集数据"""
        return self.running
    
    def stop_collection(self):
        """停止数据采集"""
        self.running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
        self.log_callback("数据采集已停止")
    
    def _trigger_training(self, app):
        """触发应用的训练功能"""
        try:
            self.log_callback("[蓝牙采集][自动训练][关键步骤] 开始调用app.start_training()...")
            # 直接调用app的start_training方法
            app.start_training()
            self.log_callback("[蓝牙采集][自动训练][关键步骤] app.start_training()调用成功")
        except Exception as e:
            self.log_callback(f"[蓝牙采集][自动训练][关键步骤] 调用app.start_training()失败: {str(e)}")
            import traceback
            self.log_callback(f"[蓝牙采集][自动训练][关键步骤] 调用失败详细错误: {traceback.format_exc()}")
            # 添加一个messagebox提示用户，以便用户知道发生了错误
            if 'messagebox' in sys.modules:
                try:
                    messagebox.showerror("自动训练失败", f"调用训练函数时出错: {str(e)}")
                except:
                    pass  # 忽略messagebox可能的错误
    
    def _start_training(self, app=None):
        """在GUI线程中开始训练，支持多通道数据处理"""
        try:
            # 如果没有传入app参数，尝试使用self.app
            if app is None and hasattr(self, 'app'):
                app = self.app
            
            # 1. 直接从蓝牙采集的数据中提取时间和电流数据
            if not hasattr(self, 'collected_data') or len(self.collected_data) == 0:
                error_msg = "未找到蓝牙采集的数据点"
                self.log_callback(error_msg)
                if 'messagebox' in sys.modules:
                    messagebox.showerror("错误", error_msg)
                return
            
            # 提取时间和多通道电流数据
            time_data = []
            multi_channel_data = {}
            max_channels = 0
            
            for point in self.collected_data:
                if isinstance(point, dict) and 'Time' in point:
                    time_data.append(float(point['Time']))
                    
                    # 优先使用channel_*字段
                    for key, value in point.items():
                        if key.startswith('channel_'):
                            try:
                                channel_num = int(key.split('_')[1])
                                if channel_num not in multi_channel_data:
                                    multi_channel_data[channel_num] = []
                                multi_channel_data[channel_num].append(float(value))
                                max_channels = max(max_channels, channel_num)
                            except (ValueError, TypeError):
                                continue
                    
                    # 如果没有channel_字段但有Data列表，处理Data列表中的通道数据
                    if 'Data' in point and isinstance(point['Data'], list):
                        for i, current in enumerate(point['Data']):
                            channel_num = i + 1
                            if channel_num not in multi_channel_data:
                                multi_channel_data[channel_num] = []
                            try:
                                multi_channel_data[channel_num].append(float(current))
                                max_channels = max(max_channels, channel_num)
                            except (ValueError, TypeError):
                                continue
            
            # 检查提取的数据是否有效
            if len(time_data) == 0 or len(multi_channel_data) == 0:
                error_msg = "无法从采集的数据中提取有效的时间-电流数据"
                self.log_callback(error_msg)
                if 'messagebox' in sys.modules:
                    messagebox.showerror("错误", error_msg)
                return
            
            # 为了兼容现有代码，使用第一个通道作为默认电流数据
            current_data = multi_channel_data.get(1, [])
            
            self.log_callback(f"成功提取 {len(time_data)} 个数据点，检测到 {len(multi_channel_data)} 个通道的数据")
            
            # 2. 保留文件路径设置作为备份
            if app and hasattr(app, 'file_var'):
                app.file_var.set(self.file_path)
                self.log_callback(f"已将蓝牙采集数据文件设置到左侧界面: {self.file_path}")
            
            # 3. 使用新添加的方法直接设置蓝牙数据到synaptic_section
            if app and hasattr(app, 'synaptic_section'):
                # 首先设置蓝牙数据
                self.log_callback("直接设置蓝牙数据到突触数据处理器...")
                success = True
                
                # 尝试使用新的set_bluetooth_multi_channel_data方法
                if hasattr(app.synaptic_section, 'set_bluetooth_multi_channel_data'):
                    self.log_callback("尝试使用多通道数据处理接口...")
                    success = app.synaptic_section.set_bluetooth_multi_channel_data(time_data, multi_channel_data)
                # 回退到单通道接口
                elif hasattr(app.synaptic_section, 'set_bluetooth_data'):
                    self.log_callback("使用单通道数据处理接口（使用第一个通道）...")
                    success = app.synaptic_section.set_bluetooth_data(time_data, current_data)
                
                if success:
                    self.log_callback("蓝牙数据设置成功")
                    # 自动启用蓝牙数据模式
                    if hasattr(app.synaptic_section, 'bluetooth_var'):
                        app.synaptic_section.bluetooth_var.set(True)
                        app.synaptic_section.use_bluetooth_data = True
                        app.synaptic_section.log_callback("自动启用蓝牙数据模式")
                    
                    # 先处理数据
                    self.log_callback("处理数据以生成归一化数据...")
                    app.synaptic_section.process_data()
                    
                    # 显示峰值检测结果和归一化数据的可视化
                    self.log_callback("显示峰值检测结果...")
                    app.synaptic_section.show_peak_detection()
                    
                    self.log_callback("显示归一化数据曲线图...")
                    app.synaptic_section.show_normalized_data()
                    
                    # 如果有多个通道，显示多通道数据可视化（如果支持）
                    if max_channels > 1 and hasattr(app.synaptic_section, 'show_multi_channel_data'):
                        self.log_callback(f"显示多通道数据可视化... (共{max_channels}个通道)")
                        app.synaptic_section.show_multi_channel_data(multi_channel_data)
                else:
                    self.log_callback("设置蓝牙数据失败，回退到原始方式")
                    # 回退到原始方式
                    app.synaptic_section.process_data()
                    app.synaptic_section.show_peak_detection()
                    app.synaptic_section.show_normalized_data()
            else:
                self.log_callback("警告：应用缺少synaptic_section组件，无法显示可视化结果")
            
            # 在GUI线程中调用应用的start_training方法
            # 这是Tkinter应用程序的正确做法，确保GUI操作在主线程中执行
            if app:
                self.log_callback("[蓝牙采集][自动训练] 检查app对象属性...")
                if hasattr(app, 'start_training'):
                    self.log_callback("[蓝牙采集][自动训练] 确认app.start_training方法存在")
                    if hasattr(app, 'root'):
                        self.log_callback("[蓝牙采集][自动训练] 确认app.root存在")
                        if hasattr(app.root, 'after'):
                            self.log_callback("[蓝牙采集][自动训练] 使用app.root.after在GUI主线程中触发_trigger_training")
                            # 使用after方法确保在GUI线程中执行
                            app.root.after(100, lambda: self._trigger_training(app))
                        else:
                            self.log_callback("[蓝牙采集][自动训练] 警告: app.root没有after方法")
                            # 作为后备方案，直接调用
                            self.log_callback("[蓝牙采集][自动训练] 直接调用_trigger_training作为后备方案")
                            self._trigger_training(app)
                    else:
                        self.log_callback("[蓝牙采集][自动训练] 警告: app没有root属性")
                        # 作为后备方案，直接调用
                        self.log_callback("[蓝牙采集][自动训练] 直接调用_trigger_training作为后备方案")
                        self._trigger_training(app)
                else:
                    self.log_callback("[蓝牙采集][自动训练] 错误: app对象缺少start_training方法")
            else:
                self.log_callback("[蓝牙采集][自动训练] 错误: app参数为None")
        except Exception as e:
            self.log_callback(f"自动训练失败: {str(e)}")
            # 显示详细错误信息以帮助调试
            import traceback
            self.log_callback(f"详细错误信息: {traceback.format_exc()}")

class BleakBluetoothDataCollector(BaseBluetoothDataCollector):
    """基于Bleak库的蓝牙数据采集器 - 支持BLE设备"""
    
   
    DEFAULT_TARGET_NAME = "CM1051"
    DEFAULT_TARGET_ADDRESS = "20:24:06:25:03:21"
    DEFAULT_PORT = 1
    
    def __init__(self, log_callback: Optional[Callable] = None, 
                 data_callback: Optional[Callable] = None):
        """初始化基于Bleak的蓝牙数据采集器"""
        super().__init__(log_callback, data_callback)
        
        # Bleak相关属性
        self.client = None
        self.loop = None
        self.connection_thread = None
        self.buffer = b''
        
        # 设置默认设备信息
        self.target_device_name = self.DEFAULT_TARGET_NAME
        self.target_device_address = self.DEFAULT_TARGET_ADDRESS
        
        # 用于数据交换的特征UUID（需要根据实际设备调整）
        self.NOTIFICATION_CHARACTERISTIC_UUID = "00002a37-0000-1000-8000-00805f9b34fb"  # 示例UUID
        
        self.log_callback(f"Bleak蓝牙数据采集器初始化完成 (BLE可用: {BLE_AVAILABLE})")
    
    def _run_async(self, coro):
        """运行异步函数并等待结果"""
        if not self.loop or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        return self.loop.run_until_complete(coro)
    
    async def _async_find_nearby_devices(self, duration: int = 8) -> List[tuple]:
        """异步搜索附近的BLE设备"""
        devices = []
        try:
            self.log_callback(f"开始搜索BLE设备（{duration}秒）...")
            
            # 扫描设备
            discovered_devices = await BleakScanner.discover(timeout=duration)
            
            for device in discovered_devices:
                devices.append((device.address, device.name or "Unknown"))
                self.log_callback(f"发现设备: {device.name or 'Unknown'} ({device.address})")
            
            self.log_callback(f"扫描完成，共发现 {len(devices)} 个BLE设备")
            return devices
            
        except Exception as e:
            self.log_callback(f"搜索BLE设备失败: {str(e)}")
            return []
    
    def find_nearby_devices(self, duration: int = 8) -> List[tuple]:
        """搜索附近的蓝牙设备"""
        if not BLE_AVAILABLE:
            self.log_callback("Bleak库不可用，无法搜索设备")
            return []
        
        return self._run_async(self._async_find_nearby_devices(duration))
    
    def find_target_device(self, target_name: str = None) -> Optional[str]:
        """查找目标蓝牙设备"""
        # 如果没有提供目标名称，使用默认值
        if target_name is None:
            target_name = self.DEFAULT_TARGET_NAME
            
        self.log_callback(f"正在查找目标设备: {target_name}")
        devices = self.find_nearby_devices()
        
        for addr, name in devices:
            if name and target_name.lower() in name.lower():
                self.target_device_name = name
                self.target_device_address = addr
                self.log_callback(f"找到目标设备: {name} ({addr})")
                return addr
        
        self.log_callback(f"未找到目标设备: {target_name}")
        return None
    
    async def _async_connect(self, address: str) -> bool:
        """异步连接到BLE设备"""
        try:
            self.log_callback(f"正在连接到设备 {address}...")
            
            # 连接到设备
            self.client = BleakClient(address)
            await self.client.connect()
            
            if self.client.is_connected:
                self.connected = True
                self.log_callback(f"成功连接到设备 {self.target_device_name or address}")
                
                # 发现设备服务和特征
                for service in self.client.services:
                    for char in service.characteristics:
                        self.log_callback(f"特征: {char.uuid} - {char.properties}")
                
                # 设置通知回调（如果支持）
                if self.NOTIFICATION_CHARACTERISTIC_UUID:
                    try:
                        await self.client.start_notify(
                            self.NOTIFICATION_CHARACTERISTIC_UUID,
                            self._notification_handler
                        )
                        self.log_callback(f"已启用通知特征 {self.NOTIFICATION_CHARACTERISTIC_UUID}")
                    except Exception as e:
                        self.log_callback(f"无法启用通知: {str(e)}")
                
                return True
            else:
                self.log_callback("设备连接失败")
                return False
                
        except Exception as e:
            self.log_callback(f"连接设备时出错: {str(e)}")
            self.connected = False
            return False
    
    def _notification_handler(self, sender, data):
        """处理接收到的通知数据"""
        try:
            # 将接收到的数据添加到缓冲区
            self.buffer += data
            
            # 检查是否有完整的数据包（假设以换行符分隔）
            while b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                line = line.decode('utf-8', errors='ignore').strip()
                if line and line.startswith('VAL'):
                    self.log_callback(f"收到通知数据: {line}")
                    # 解析数据并处理
                    data_point = self.parse_data_line(line)
                    if data_point and self.running:
                        # 这里可以添加数据处理逻辑
                        pass
                
        except Exception as e:
            self.log_callback(f"处理通知数据时出错: {str(e)}")
    
    def connect_target_device(self, target_name: str = None, target_address: str = None, port: int = None) -> bool:
        """连接到目标设备
        
        使用Python内置socket库连接到已配对的蓝牙设备
        """
        self.log_callback("开始连接蓝牙设备...")
        
        # 设置目标参数
        if target_name is None:
            target_name = self.DEFAULT_TARGET_NAME
        if port is None:
            port = self.DEFAULT_PORT
        
        self.device_name = target_name
        self.port = port
        
        # 使用提供的地址或查找设备
        if not target_address:
            target_address = self.find_target_device(target_name)
            if not target_address:
                self.log_callback("无法找到目标设备")
                return False
        else:
            self.device_address = target_address
        
        try:
            # 创建RFCOMM套接字连接
            self.sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
            # 设置套接字超时
            self.sock.settimeout(5.0)
            self.log_callback(f"正在连接到 {self.device_name} ({target_address}:{port})...")
            
            # 连接到设备
            self.sock.connect((target_address, port))
            self.connected = True
            self.is_connected = True  # 同步更新两个属性
            self.log_callback(f"成功连接到设备: {self.device_name}")
            
            # 发送开始采集命令到设备
            try:
                self.sock.sendall(b"START\n")
                self.log_callback("已发送开始采集命令到设备")
            except Exception as e:
                self.log_callback(f"发送命令失败: {str(e)}")
                
            return True
            
        except Exception as e:
            self.log_callback(f"连接设备时出错: {str(e)}")
            self.connected = False
            self.is_connected = False
            self.sock = None
            
            # 如果连接失败，尝试使用默认地址的备用方案
            if target_address != self.DEFAULT_TARGET_ADDRESS and self.DEFAULT_TARGET_ADDRESS:
                self.log_callback(f"尝试使用默认地址备用连接: {self.DEFAULT_TARGET_ADDRESS}")
                try:
                    self.sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
                    self.sock.settimeout(5.0)
                    self.sock.connect((self.DEFAULT_TARGET_ADDRESS, port))
                    self.connected = True
                    self.device_address = self.DEFAULT_TARGET_ADDRESS
                    self.log_callback(f"成功使用默认地址连接到设备")
                    return True
                except Exception as e2:
                    self.log_callback(f"备用连接也失败: {str(e2)}")
                    self.connected = False
                    self.sock = None
            
            return False
            
    def disconnect(self):
        """断开与设备的连接"""
        try:
            # 停止数据采集
            self.running = False
            
            # 关闭套接字连接
            if self.sock:
                self.sock.close()
                self.sock = None
            
            # 更新连接状态
            self.connected = False
            self.is_connected = False
            self.log_callback("已断开与设备的连接")
            return True
        except Exception as e:
            self.log_callback(f"断开连接时出错: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """检查是否已连接设备"""
        if not BLE_AVAILABLE:
            return False
        
        return self.connected and self.client and self.client.is_connected
    
    async def _async_read_data(self) -> Optional[str]:
        """异步从设备读取数据"""
        if not self.is_connected():
            self.log_callback("设备未连接")
            return None
        
        try:
            # 如果有读取特征UUID，可以直接读取
            # 这里需要根据实际设备设置正确的特征UUID
            # 例如：value = await self.client.read_gatt_char(READ_CHARACTERISTIC_UUID)
            
            # 注意：这里需要根据实际设备的特性修改，以下是示例代码
            # 实际应用中请替换为适合您设备的读取方式
            self.log_callback("读取数据：请根据实际设备修改_read_data方法")
            # 暂时返回None，等待设备连接后实现真实读取
            return None
            
        except Exception as e:
            self.log_callback(f"读取数据时出错: {str(e)}")
            self.connected = False
            return None
    
    def read_data(self, timeout: float = 1.0) -> Optional[str]:
        """从蓝牙设备读取数据"""
        if not BLE_AVAILABLE:
            self.log_callback("Bleak库不可用，无法读取数据")
            return None
        
        return self._run_async(self._async_read_data())
    
    def collect_data(self, target_points: int = -1, auto_train: bool = False, app=None):
        """开始数据采集
        
        Args:
            target_points: 目标数据点数，-1表示持续采集
            auto_train: 是否自动触发训练
            app: 应用实例
        """
        if not self.is_connected():
            self.log_callback("请先连接蓝牙设备")
            return
        
        self.target_points = target_points
        self.running = True
        
        def collection_worker():
            """数据采集工作线程"""
            collected_points = 0
            self.collected_data = []
            
            if target_points > 0:
                self.log_callback(f"开始采集 {target_points} 个数据...")
            else:
                self.log_callback("开始持续采集数据...")
            
            while self.running and (target_points <= 0 or collected_points < target_points):
                try:
                    # 读取数据
                    data_line = self.read_data(timeout=2.0)
                    if not data_line:
                        continue
                    
                    # 解析数据
                    data_point = self.parse_data_line(data_line)
                    if not data_point:
                        continue
                    
                    # 添加到收集的数据
                    self.collected_data.append(data_point)
                    collected_points += 1
                    
                    # 数据回调
                    if self.data_callback:
                        callback_data = {
                            'data_point': data_point,
                            'collected_points': collected_points,
                            'target_points': target_points
                        }
                        if target_points > 0:
                            callback_data['progress'] = collected_points / target_points
                        self.data_callback(callback_data)
                    
                    # 日志输出
                    current_str = ", ".join([self.get_current_unit(val) for val in data_point['Data']])
                    if target_points > 0:
                        self.log_callback(f"已采集 {collected_points}/{target_points}: "
                                        f"时间={data_point['Time']:.3f}s, 电流=[{current_str}]")
                    else:
                        self.log_callback(f"已采集 {collected_points} 个: "
                                        f"时间={data_point['Time']:.3f}s, 电流=[{current_str}]")
                    
                    # 更新进度
                    if self.progress_callback:
                        self.progress_callback(collected_points, target_points)
                    
                    # 短暂延时，避免过快采集
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.log_callback(f"数据采集出错: {str(e)}")
                    break
            
            # 采集完成
            self.running = False
            
            self.log_callback(f"数据采集完成！共采集 {collected_points} 个数据")
            
            # 保存数据
            if collected_points > 0:
                success = self.save_to_csv(self.collected_data)
                
                # 仅在非持续采集模式下才触发自动训练
                if success and auto_train and app and target_points > 0:
                    # 自动开始训练
                    self.log_callback("开始自动训练...")
                    
                    # 确保使用正确的数据文件
                    app.file_path = self.file_path
                    
                    # 在GUI线程中开始训练
                    if hasattr(app, 'root'):
                        app.root.after(1000, lambda: self._start_training(app))
            else:
                self.log_callback("没有数据需要保存")
        
        # 启动采集线程
        self.collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self.collection_thread.start()
    
    async def _async_disconnect(self):
        """异步断开连接"""
        if self.client:
            try:
                # 停止通知
                if self.NOTIFICATION_CHARACTERISTIC_UUID:
                    try:
                        await self.client.stop_notify(self.NOTIFICATION_CHARACTERISTIC_UUID)
                    except:
                        pass
                
                # 断开连接
                await self.client.disconnect()
                self.log_callback("BLE设备连接已断开")
            except Exception as e:
                self.log_callback(f"断开连接时出错: {str(e)}")
            finally:
                self.client = None
                self.connected = False
    
    def disconnect(self):
        """断开蓝牙连接"""
        self.running = False
        
        if BLE_AVAILABLE:
            try:
                self._run_async(self._async_disconnect())
            except Exception as e:
                self.log_callback(f"执行断开连接时出错: {str(e)}")
        else:
            self.connected = False
            self.log_callback("模拟断开连接")




class WindowsBluetoothDataCollector(BaseBluetoothDataCollector):
    """使用Windows原生方法连接已配对蓝牙设备的数据采集器"""
    
    DEFAULT_TARGET_NAME = "CM1051"
    DEFAULT_TARGET_ADDRESS = "20:24:06:25:03:21"
    DEFAULT_PORT = 1
    
    def __init__(self, log_callback: Optional[Callable] = None, 
                 data_callback: Optional[Callable] = None):
        super().__init__(log_callback, data_callback)
        self.sock = None
        self.connected = False
        self.running = False
        self.device_address = None
        self.device_name = None
        self.port = None
        self.collection_thread = None
    
    def is_connected(self) -> bool:
        """检查是否已连接到设备"""
        return self.connected
        
    def disconnect(self):
        """断开与设备的连接"""
        try:
            # 停止数据采集
            self.running = False
            
            # 关闭套接字连接
            if self.sock:
                self.sock.close()
                self.sock = None
            
            # 更新连接状态
            self.connected = False
            self.log_callback("已断开与设备的连接")
            return True
        except Exception as e:
            self.log_callback(f"断开连接时出错: {str(e)}")
            return False
    
    def find_target_device(self, target_name: str = None) -> Optional[str]:
        """查找目标设备地址
        
        使用Windows命令行工具查找已配对设备
        """
        if target_name is None:
            target_name = self.DEFAULT_TARGET_NAME
        
        self.log_callback(f"查找设备: {target_name}...")
        
        try:
            # 在Windows上使用powershell命令获取已配对蓝牙设备
            if sys.platform == 'win32':
                # 使用PowerShell命令获取已配对的蓝牙设备
                cmd = "powershell -Command \"Get-PnpDevice -Class Bluetooth | Where-Object {$_.Status -eq 'OK'} | Format-List Name,InstanceId\""
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                # 解析输出以查找设备名称和地址
                output = result.stdout
                self.log_callback(f"PowerShell输出: {output}")
                
                # 尝试从输出中提取设备信息
                lines = output.strip().split('\n')
                current_name = None
                current_id = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Name:'):
                        current_name = line[5:].strip()
                        self.log_callback(f"发现设备名称: {current_name}")
                    elif line.startswith('InstanceId:'):
                        current_id = line[12:].strip()
                        # 从InstanceId中提取MAC地址（格式通常为BTHENUM\DEV_XXXX_XX_XX_XX_XX_XX）
                        mac_match = re.search(r'DEV_([0-9A-F]{12})', current_id)
                        if mac_match:
                            mac_raw = mac_match.group(1)
                            # 格式化为标准MAC地址格式: XX:XX:XX:XX:XX:XX
                            mac_address = ':'.join([mac_raw[i:i+2] for i in range(0, 12, 2)]).upper()
                            self.log_callback(f"已配对设备: {current_name} ({mac_address})")
                            
                            # 检查是否是目标设备
                            if current_name and target_name.lower() in current_name.lower():
                                self.device_address = mac_address
                                self.device_name = current_name
                                self.log_callback(f"找到目标设备: {current_name} ({mac_address})")
                                return mac_address
            
            # 如果没有找到或不是Windows系统，尝试直接使用默认地址
            if self.DEFAULT_TARGET_ADDRESS:
                self.log_callback(f"未在已配对设备中找到，尝试使用默认地址: {self.DEFAULT_TARGET_ADDRESS}")
                self.device_address = self.DEFAULT_TARGET_ADDRESS
                self.device_name = target_name
                return self.DEFAULT_TARGET_ADDRESS
                
            self.log_callback(f"未找到设备: {target_name}")
            return None
            
        except Exception as e:
            self.log_callback(f"查找设备时出错: {str(e)}")
            # 如果出错，尝试直接使用默认地址
            if self.DEFAULT_TARGET_ADDRESS:
                self.log_callback(f"查找出错，尝试使用默认地址: {self.DEFAULT_TARGET_ADDRESS}")
                self.device_address = self.DEFAULT_TARGET_ADDRESS
                self.device_name = target_name
                return self.DEFAULT_TARGET_ADDRESS
            return None
    
    def connect_target_device(self, target_name: str = None, target_address: str = None, port: int = None) -> bool:
        """连接到目标设备
        
        使用Python内置socket库连接到已配对的蓝牙设备
        """
        self.log_callback("开始连接蓝牙设备...")
        
        # 设置目标参数
        if target_name is None:
            target_name = self.DEFAULT_TARGET_NAME
        if port is None:
            port = self.DEFAULT_PORT
        
        self.device_name = target_name
        self.port = port
        
        # 使用提供的地址或查找设备
        if not target_address:
            target_address = self.find_target_device(target_name)
            if not target_address:
                self.log_callback("无法找到目标设备")
                return False
        else:
            self.device_address = target_address
        
        try:
            # 创建RFCOMM套接字连接
            self.sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
            # 设置套接字超时
            self.sock.settimeout(5.0)
            self.log_callback(f"正在连接到 {self.device_name} ({target_address}:{port})...")
            
            # 连接到设备
            self.sock.connect((target_address, port))
            self.connected = True
            self.log_callback(f"成功连接到设备: {self.device_name}")
            
            # 发送开始采集命令到设备
            try:
                self.sock.sendall(b"START\n")
                self.log_callback("已发送开始采集命令到设备")
            except Exception as e:
                self.log_callback(f"发送命令失败: {str(e)}")
                
            return True
            
        except Exception as e:
            self.log_callback(f"连接设备时出错: {str(e)}")
            self.connected = False
            self.sock = None
            
            # 如果连接失败，尝试使用默认地址的备用方案
            if target_address != self.DEFAULT_TARGET_ADDRESS and self.DEFAULT_TARGET_ADDRESS:
                self.log_callback(f"尝试使用默认地址备用连接: {self.DEFAULT_TARGET_ADDRESS}")
                try:
                    self.sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
                    self.sock.settimeout(5.0)
                    self.sock.connect((self.DEFAULT_TARGET_ADDRESS, port))
                    self.connected = True
                    self.is_connected = True
                    self.device_address = self.DEFAULT_TARGET_ADDRESS
                    self.log_callback(f"成功使用默认地址连接到设备")
                    return True
                except Exception as e2:
                    self.log_callback(f"备用连接也失败: {str(e2)}")
                    self.connected = False
                    self.is_connected = False
                    self.sock = None
            
            return False
    
    def collect_data(self, target_points: int = 1000, auto_train: bool = False, app: object = None) -> bool:
        """开始从已连接的设备收集数据"""
        if not self.connected or not self.sock:
            self.log_callback("未连接到设备，无法开始采集")
            return False
        
        self.running = True
        self.collected_data = []
        collected_points = 0
        self.start_time = time.time()  # 初始化start_time，用于计算相对时间
        buffer = b""  # 初始化缓冲区
        
        def collection_worker():
            nonlocal collected_points, buffer
            try:
                # 根据采集模式显示不同的日志
                if target_points > 0:
                    self.log_callback(f"开始采集数据，目标 {target_points} 个数据点")
                else:
                    self.log_callback("开始持续采集数据...")
                
                # 持续采集模式：target_points <= 0 时条件永远为True，直到self.running变为False
                while self.running and (target_points <= 0 or collected_points < target_points):
                    try:
                        # 设置超时，避免阻塞
                        self.sock.settimeout(2.0)
                        
                        # 接收数据
                        data = self.sock.recv(1024)
                        
                        if not data:
                            self.log_callback("未收到数据，连接可能已断开")
                            break
                        
                        # 将接收到的数据添加到缓冲区
                        buffer += data
                        
                        # 处理缓冲区中的完整行（参考Save.py的实现）
                        while b'\n' in buffer:
                            # 提取单行并保留剩余部分
                            line, buffer = buffer.split(b'\n', 1)
                            line = line.decode('utf-8', errors='ignore').strip()
                            
                            # 只处理非空且以VAL开头的行
                            if line and line.startswith('VAL'):
                                self.log_callback(f"收到原始数据: {line}")
                                 
                                try:
                                    data_point = self.parse_data_line(line)
                                    if not data_point:
                                        self.log_callback(f"无法解析数据行: {line}")
                                        continue
                                     
                                    # 添加到收集的数据
                                    self.collected_data.append(data_point)
                                    collected_points += 1
                                     
                                    # 数据回调
                                    if self.data_callback:
                                        callback_data = {
                                            'data_point': data_point,
                                            'collected_points': collected_points,
                                            'target_points': target_points
                                        }
                                        # 仅在非持续采集模式下计算进度
                                        if target_points > 0:
                                            callback_data['progress'] = collected_points / target_points
                                        self.data_callback(callback_data)
                                     
                                    # 日志输出 - 支持多通道数据显示
                                    # 格式化电流显示（根据Save.py的单位处理）
                                    current_displays = []
                                    for current_val in data_point['Data']:
                                        # 根据电流值大小选择合适的显示单位
                                        if abs(current_val) >= 1e-3:
                                            current_displays.append(f"{current_val * 1000:.3f}mA")
                                        elif abs(current_val) >= 1e-6:
                                            current_displays.append(f"{current_val * 1000000:.3f}μA")
                                        elif abs(current_val) >= 1e-9:
                                            current_displays.append(f"{current_val * 1000000000:.3f}nA")
                                        else:
                                            current_displays.append(f"{current_val * 1000000000000:.3f}pA")
                                     
                                    current_str = ", ".join(current_displays)
                                    # 根据采集模式显示不同的日志格式
                                    if target_points > 0:
                                        self.log_callback(f"已采集 {collected_points}/{target_points}: "
                                                        f"时间={data_point['Time']:.3f}s, "
                                                        f"通道={data_point.get('Channel', 'N/A')}, "
                                                        f"电流=[{current_str}]")
                                    else:
                                        self.log_callback(f"已采集 {collected_points}: "
                                                        f"时间={data_point['Time']:.3f}s, "
                                                        f"通道={data_point.get('Channel', 'N/A')}, "
                                                        f"电流=[{current_str}]")
                                     
                                    # 更新进度
                                    if self.progress_callback:
                                        self.progress_callback(collected_points, target_points)
                                except Exception as e:
                                    self.log_callback(f"处理单条数据时出错: {str(e)}")
                                    # 继续处理下一条数据，不中断整个采集过程
                            
                            # 短暂延时，避免过快采集
                            time.sleep(0.1)
                            
                            # 检查是否达到目标（仅在非持续采集模式下）
                            if target_points > 0 and collected_points >= target_points:
                                break
                    
                    except socket.timeout:
                        # 超时是正常的，继续尝试接收
                        current_time = time.time()
                        # 如果长时间没有收到数据，发送一个心跳命令
                        if current_time - self.start_time > 5.0 and collected_points == 0:
                            self.log_callback("长时间未收到数据，尝试发送心跳命令")
                            try:
                                self.sock.sendall(b"SYNC\n")
                            except:
                                pass
                        continue
                    except Exception as e:
                        self.log_callback(f"数据采集出错: {str(e)}")
                        # 记录错误但继续采集，不要立即中断
                        time.sleep(0.5)  # 短暂暂停后继续尝试
            
            finally:
                # 采集完成
                self.running = False
                
                self.log_callback(f"数据采集完成！共采集 {collected_points} 个数据")
                
                # 保存数据
                if collected_points > 0:
                    success = self.save_to_csv(self.collected_data)
                    
                    # 仅在非持续采集模式下才触发自动训练
                    if success and auto_train and app and target_points > 0:
                        # 自动开始训练
                        self.log_callback("[蓝牙采集][自动训练] 开始自动训练流程...")
                        
                        # 确保使用正确的数据文件
                        if hasattr(app, 'file_path'):
                            app.file_path = self.file_path
                            self.log_callback(f"[蓝牙采集][自动训练] 已设置数据文件路径: {self.file_path}")
                        else:
                            self.log_callback("[蓝牙采集][自动训练] 警告: app对象没有file_path属性")
                        
                        # 在GUI线程中开始训练
                        if hasattr(app, 'root'):
                            self.log_callback("[蓝牙采集][自动训练] 准备在GUI主线程中调用_start_training方法")
                            app.root.after(1000, lambda: self._start_training(app))
                        else:
                            self.log_callback("[蓝牙采集][自动训练] 警告: app对象没有root属性，无法在GUI线程中执行")
                            # 直接调用作为后备方案
                            self._start_training(app)
                else:
                    self.log_callback("没有数据需要保存")
        
        # 启动采集线程
        self.collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self.collection_thread.start()
        return True

# 选择合适的蓝牙数据采集器实现
def get_available_collector():
    """获取可用的蓝牙数据采集器类"""
    # 优先使用Windows原生实现（用于已配对设备）
    if sys.platform == 'win32':
        return WindowsBluetoothDataCollector
    # 否则使用Bleak实现
    elif BLE_AVAILABLE:
        return BleakBluetoothDataCollector
    # 如果都不可用，返回基础类（功能受限）
    else:
        return BaseBluetoothDataCollector

# 设置默认的蓝牙数据采集器
BluetoothDataCollector = get_available_collector()


# 全局实例和获取函数
_bluetooth_collector_instance = None

def get_bluetooth_collector(log_callback: Optional[Callable] = None, 
                           data_callback: Optional[Callable] = None) -> BluetoothDataCollector:
    """获取蓝牙数据采集器单例实例"""
    global _bluetooth_collector_instance
    
    if _bluetooth_collector_instance is None:
        _bluetooth_collector_instance = BluetoothDataCollector(
            log_callback=log_callback,
            data_callback=data_callback
        )
    
    # 更新回调函数（如果提供了新的）
    if log_callback:
        _bluetooth_collector_instance.log_callback = log_callback
    if data_callback:
        _bluetooth_collector_instance.data_callback = data_callback
    
    return _bluetooth_collector_instance

# 添加蓝牙库安装提示
def install_bluetooth_hint():
    """返回安装蓝牙库的提示信息"""
    return """要启用完整的蓝牙功能：
        
        1. Windows系统上：
           - 程序将自动使用系统已配对设备
           - 确保您的设备已在Windows蓝牙设置中完成配对
        
        2. 对于BLE设备连接（可选）：
           pip install bleak
        
        对于BLE设备连接，您可能需要修改bluetooth_data_collector.py中的特征UUID
        以匹配您的实际设备。"""

# 保留原有函数以保持兼容性
def install_bleak_hint():
    """返回安装Bleak库的提示信息（兼容旧代码）"""
    return install_bluetooth_hint()

# 蓝牙库可用性检查已静默处理，不显示状态信息