import pyvista as pv
import numpy as np
import os
import requests
import json
import rasterio
from rasterio.enums import Resampling

def fetch_opentopography_dem(bbox, api_key, dataset="SRTMGL1", output_path="shenzhen_dem.tif"):
    """
    从 OpenTopography 下载 DEM 数据。
    
    参数:
    - bbox: [south, west, north, east] 经纬度范围
    - api_key: OpenTopography 的 API Key
    - dataset: 数据集名称，默认为 SRTMGL1 (30m 分辨率)
    """
    if os.path.exists(output_path):
        print(f"检测到本地 DEM 数据: {output_path}")
        return output_path

    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": dataset,
        "south": bbox[0],
        "west": bbox[1],
        "north": bbox[2],
        "east": bbox[3],
        "outputFormat": "GTiff",
        "API_Key": api_key
    }

    print(f"正在从 OpenTopography 下载 {dataset} 数据...")
    try:
        response = requests.get(url, params=params, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"下载成功: {output_path}")
        return output_path
    except Exception as e:
        print(f"下载 DEM 失败: {e}")
        return None

def load_dem_to_grid(dem_path):
    """使用 rasterio 加载 GeoTIFF 并转换为 PyVista 网格"""
    if not dem_path or not os.path.exists(dem_path):
        return None, None, None

    with rasterio.open(dem_path) as src:
        # 读取第一波段（高程值）
        elevation = src.read(1)
        # 获取地理范围和分辨率
        bounds = src.bounds
        res_x = (bounds.right - bounds.left) / src.width
        res_y = (bounds.top - bounds.bottom) / src.height
        origin = (bounds.left, bounds.bottom, 0)
        spacing = (res_x, res_y, 1)
        
        # 处理无效值 (NoData)
        elevation = elevation.astype(float)
        elevation[elevation == src.nodata] = 0
        
        # rasterio 读取的数据通常是 (North -> South)，需要翻转以匹配 PyVista 坐标系
        elevation = np.flipud(elevation)
        
        return elevation, spacing, origin

def fetch_shenzhen_geojson(full=True):
    """
    从阿里云 DataV 下载深圳市行政区边界 GeoJSON。
    
    参数:
    - full: True 表示包含各区边界，False 仅包含全市外轮廓
    """
    adcode = "440300"
    suffix = "_full" if full else ""
    url = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}{suffix}.json"
    local_path = f"shenzhen_bound{suffix}.json"
    
    if os.path.exists(local_path):
        print(f"检测到本地 GeoJSON: {local_path}")
        with open(local_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    print(f"正在从 {url} 下载深圳市边界数据...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # 保存到本地
        with open(local_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return data
    except Exception as e:
        print(f"下载失败: {e}")
        return None

def parse_geojson_to_polydata(geojson_data):
    """将 GeoJSON 坐标转换为 PyVista 的线条对象"""
    if not geojson_data: return None
    
    all_points = []
    all_lines = []
    offset = 0
    
    for feature in geojson_data['features']:
        geom = feature['geometry']
        # 处理 Polygon 和 MultiPolygon
        polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        
        for poly in polys:
            for ring in poly:
                # ring 是 [[lon1, lat1], [lon2, lat2], ...]
                # 为了在沙盒中显示，我们假设经纬度直接映射到 X-Y 轴
                # 实际应用中需要投影变换（如墨卡托），这里简化处理
                pts = np.array(ring)
                z = np.zeros((pts.shape[0], 1))
                pts_3d = np.hstack((pts, z))
                
                n_pts = len(pts_3d)
                all_points.extend(pts_3d)
                # 线条单元格式: [点数, 索引1, 索引2, ..., 索引n]
                line = [n_pts] + list(range(offset, offset + n_pts))
                all_lines.append(line)
                offset += n_pts
                
    if not all_points: return None
    return pv.PolyData(np.array(all_points), lines=np.concatenate(all_lines))

def create_shenzhen_sandbox(dem_path=None, satellite_path=None, api_key=None):
    """
    创建一个深圳市 3D 城市沙盒。
    """
    print("="*40)
    print("      深圳市 3D 城市沙盒渲染引擎")
    print("="*40)

    # 深圳市大致范围: [south, west, north, east]
    SZ_BBOX = [22.45, 113.75, 22.85, 114.65]

    # 1. 获取行政区划边界
    sz_geojson = fetch_shenzhen_geojson(full=True)
    border_poly = parse_geojson_to_polydata(sz_geojson)

    # 2. 获取地形数据 (优先从本地或 OpenTopography 获取)
    elevation, spacing, origin = None, None, None
    dem_file = "shenzhen_dem.tif"
    
    # 检查本地是否已有数据，或者是否有 API Key 可以下载
    if os.path.exists(dem_file):
        print(f"直接从本地读取地形数据: {dem_file}")
        elevation, spacing, origin = load_dem_to_grid(dem_file)
    elif api_key:
        dem_file = fetch_opentopography_dem(SZ_BBOX, api_key, output_path=dem_file)
        if dem_file:
            elevation, spacing, origin = load_dem_to_grid(dem_file)

    # 如果既没有本地文件，也没有 API Key，则使用模拟地形
    if elevation is None:
        print("未检测到本地数据且未提供 API Key，正在生成模拟地形...")
        nx, ny = 500, 300
        lon = np.linspace(SZ_BBOX[1], SZ_BBOX[3], nx)
        lat = np.linspace(SZ_BBOX[0], SZ_BBOX[2], ny)
        xx, yy = np.meshgrid(lon, lat)
        
        z = 0.8 * np.exp(-((xx-114.2)**2 + (yy-22.58)**2)/0.01) + \
            0.3 * np.exp(-((xx-113.9)**2 + (yy-22.48)**2)/0.005) + \
            0.05 * np.random.normal(size=xx.shape) * 0.1
        elevation = z
        spacing = (lon[1]-lon[0], lat[1]-lat[0], 1)
        origin = (lon[0], lat[0], 0)

    # 3. 创建 PyVista 网格对象
    grid = pv.ImageData(dimensions=(elevation.shape[1], elevation.shape[0], 1), spacing=spacing, origin=origin)
    grid.point_data["Elevation"] = elevation.flatten(order="C")

    # 4. 设置可视化
    pv.global_theme.multi_samples = 0
    plotter = pv.Plotter(title="深圳市 3D 城市沙盒")
    plotter.enable_anti_aliasing('ssaa')
    plotter.set_background("#1a1a1a")

    # 5. 地形位移 (Warping)
    # 核心：物理比例校正
    # 水平单位是度 (degree)，垂直单位是米 (meter)
    # 1度纬度约等于 111,320米。为了实现 1:1 物理比例，
    # 转换因子应为 1 / 111320.0
    DEGREE_TO_METERS = 111320.0
    base_warp = 1.0 / DEGREE_TO_METERS
    
    # 如果是模拟数据，其高度已经是 0~1 的模拟值，不需要这么小的缩放
    if elevation.max() < 10: # 模拟数据的 z 值通常很小
        base_warp = 0.1
        
    terrain_mesh = grid.warp_by_scalar("Elevation", factor=base_warp)
    
    # 6. 添加边界线
    if border_poly:
        # 将边界线固定在海拔 0 位置
        border_poly.points[:, 2] = 0
        plotter.add_mesh(border_poly, color="yellow", line_width=2, name="border", label="深圳市行政区划")

    # 7. 应用贴图
    if satellite_path and os.path.exists(satellite_path):
        texture = pv.read_texture(satellite_path)
        plotter.add_mesh(terrain_mesh, texture=texture, smooth_shading=True, name="terrain")
    else:
        plotter.add_mesh(terrain_mesh, scalars="Elevation", cmap="terrain", 
                        smooth_shading=True, name="terrain")

    # 8. 滑杆控件：调节地形夸张比例
    def update_warp(value):
        # value=1.0 时为真实比例 (1:1)
        factor = value * base_warp
        new_mesh = grid.warp_by_scalar("Elevation", factor=factor)
        terrain_mesh.points[:] = new_mesh.points
        plotter.render()

    plotter.add_slider_widget(
        callback=update_warp,
        rng=[1.0, 100.0], # 按照用户要求，将范围扩展为 1 到 100
        value=1.0,
        title="地形夸张比例 (1:N)",
        pointa=(0.7, 0.1),
        pointb=(0.95, 0.1),
        style='modern'
    )

    plotter.add_axes()
    plotter.show()

if __name__ == "__main__":
    # 在此填入你的 OpenTopography API Key
    MY_API_KEY = "62099c445180ceda66bd0b07bc905113"
    
    create_shenzhen_sandbox(api_key=MY_API_KEY)
