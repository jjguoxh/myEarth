import pyvista as pv
import numpy as np
from pyvista import examples
import sys
import json
import requests

def download_world_borders():
    """下载并解析全球国界线 GeoJSON 数据"""
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    print(f"正在从 {url} 下载全球国界线数据...")
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        all_lines_pts = []
        all_lines_cells = []
        curr_pt_idx = 0
        
        for feature in data['features']:
            geom = feature['geometry']
            coords = geom['coordinates']
            
            # 处理 Polygon 和 MultiPolygon
            polys = [coords] if geom['type'] == 'Polygon' else coords
            
            for poly in polys:
                for ring in poly:
                    # ring 是 [[lon1, lat1], [lon2, lat2], ...]
                    ring_pts = []
                    for lon, lat in ring:
                        # 转换为弧度
                        phi = np.deg2rad(lat)
                        theta = np.deg2rad(lon)
                        # 转换为 3D 笛卡尔坐标 (半径为 1.0)
                        x = np.cos(phi) * np.cos(theta)
                        y = np.cos(phi) * np.sin(theta)
                        z = np.sin(phi)
                        ring_pts.append([x, y, z])
                    
                    num_pts = len(ring_pts)
                    if num_pts < 2: continue
                    
                    all_lines_pts.extend(ring_pts)
                    # 定义线条单元: [n_pts, i1, i2, ..., in]
                    cell = [num_pts] + list(range(curr_pt_idx, curr_pt_idx + num_pts))
                    all_lines_cells.append(cell)
                    curr_pt_idx += num_pts
                    
        if not all_lines_pts:
            return None
            
        borders = pv.PolyData(np.array(all_lines_pts), lines=np.hstack(all_lines_cells))
        return borders
    except Exception as e:
        print(f"加载国界线数据失败: {e}")
        return None

def create_virtual_globe():
    print("正在加载地球地形数据和轮廓...")
    
    # 1. 加载地形网格
    try:
        topo = examples.download_topo_global()
    except Exception as e:
        print(f"下载地形数据失败: {e}")
        topo = pv.Sphere(radius=1.0, theta_resolution=120, phi_resolution=120)
        topo.point_data['altitude'] = np.zeros(topo.n_points)

    # 2. 加载地表贴图、海岸线和国界线
    try:
        texture = examples.load_globe_texture()
    except Exception as e:
        print(f"加载贴图失败: {e}")
        texture = None

    try:
        coastlines = examples.download_coastlines()
    except Exception as e:
        print(f"加载海岸线失败: {e}")
        coastlines = None
        
    borders = download_world_borders()

    # 3. 投影校正 (UV 映射)
    print("正在进行投影校正 (UV 映射)...")
    pts_orig = topo.points.copy()
    r = np.linalg.norm(pts_orig, axis=1)
    lat = np.arcsin(pts_orig[:, 2] / r)
    lon = np.arctan2(pts_orig[:, 1], pts_orig[:, 0])
    u = (lon + np.pi) / (2 * np.pi)
    v = (lat + np.pi / 2) / np.pi
    topo.active_texture_coordinates = np.column_stack((u, v))

    # 4. 准备地形位移数据
    scalars = topo.point_data['altitude']
    # 计算指向球心外的单位向量
    unit_vectors = pts_orig / r[:, np.newaxis]
    
    # 保存海岸线和国界线原始位置
    if coastlines:
        coast_pts_orig = coastlines.points.copy()
    if borders:
        borders_pts_orig = borders.points.copy()

    # 5. 设置可视化界面
    pv.global_theme.multi_samples = 0
    plotter = pv.Plotter(title="Python 3D 虚拟地形仪 - 交互式夸张比例")
    plotter.enable_anti_aliasing('ssaa')
    plotter.set_background("black")

    # 初始添加模型
    earth_actor = None
    if texture:
        earth_actor = plotter.add_mesh(topo, texture=texture, smooth_shading=True, name="earth", specular=0.1, ambient=0.3)
    else:
        earth_actor = plotter.add_mesh(topo, scalars='altitude', cmap="terrain", smooth_shading=True, name="earth", clim=[-8000, 8848])

    coast_actor = None
    if coastlines:
        coast_actor = plotter.add_mesh(coastlines, color="white", line_width=2, name="coastlines", label="海岸线")
        
    borders_actor = None
    if borders:
        borders_actor = plotter.add_mesh(borders, color="cyan", line_width=1, name="borders", label="国界线")

    # 6. 滑杆回调函数：调节地形夸张比例
    def update_exaggeration(value):
        # value 范围 1.0 到 5.0
        # 基础夸大倍数
        base_factor = 40.0
        current_factor = value * base_factor
        
        displacement = (scalars / 6371000.0) * current_factor
        topo.points = pts_orig + unit_vectors * displacement[:, np.newaxis]
        
        # 轮廓线和国界线同步浮动，保持在地形上方
        height_offset = 1.0 + (value * 0.005)
        if coastlines and coast_actor:
            coastlines.points = coast_pts_orig * height_offset
        if borders and borders_actor:
            borders.points = borders_pts_orig * (height_offset + 0.001) # 略高于海岸线以防重叠
            
        plotter.render()

    # 添加滑杆控件
    plotter.add_slider_widget(
        callback=update_exaggeration,
        rng=[1.0, 5.0],
        value=1.0,
        title="地形夸张比例 (1:N)",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
        color='white'
    )

    # 7. 交互说明
    def toggle_coastlines():
        if coast_actor:
            visible = not coast_actor.GetVisibility()
            coast_actor.SetVisibility(visible)
            
    def toggle_borders():
        if borders_actor:
            visible = not borders_actor.GetVisibility()
            borders_actor.SetVisibility(visible)

    def reset_view():
        plotter.camera_position = [(3.5, 3.5, 3.5), (0, 0, 0), (0, 0, 1)]

    plotter.add_key_event("c", toggle_coastlines)
    plotter.add_key_event("b", toggle_borders)
    plotter.add_key_event("r", reset_view)

    print("\n" + "="*30)
    print("      虚拟地球仪已启动")
    print("="*30)
    print("操作说明:")
    print("- [滑杆]       : 调节地形夸张比例 (1:1 ~ 1:5)")
    print("- [鼠标左键]   : 自由旋转")
    print("- [键盘 'C']   : 开关大洲轮廓线 (Coastlines)")
    print("- [键盘 'B']   : 开关国界线 (Borders)")
    print("- [键盘 'R']   : 重置初始视角")
    print("- [键盘 'Q']   : 退出程序")
    print("="*30)

    plotter.camera_position = [(3.5, 3.5, 3.5), (0, 0, 0), (0, 0, 1)]
    plotter.show()

if __name__ == "__main__":
    create_virtual_globe()
