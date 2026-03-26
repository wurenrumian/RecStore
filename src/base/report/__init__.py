import ctypes
import os

# 获取当前文件所在目录作为库路径
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_PACKAGE_DIR, "libreport.so")

try:
    _lib = ctypes.CDLL(_LIB_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load shared library from {_LIB_PATH}. Error: {e}")

# 定义 C 函数参数和返回类型
_lib.report.argtypes = [
    ctypes.c_char_p,  # table_name (const char*)
    ctypes.c_char_p,  # unique_id (const char*)
    ctypes.c_char_p,  # metric_name (const char*)
    ctypes.c_double   # metric_value (double)
]
_lib.report.restype = ctypes.c_bool

def report_metric(table: str, uid: str, metric: str, value: float) -> bool:
    """
    Report a metric to the backend service.
    
    Args:
        table (str): Target table name (e.g., 'py_test_results').
        uid (str): Unique identifier for the data row.
        metric (str): Metric column name.
        value (float): Metric value.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    return _lib.report(
        table.encode('utf-8'),
        uid.encode('utf-8'),
        metric.encode('utf-8'),
        float(value)
    )
