import pyvista as pv
import numpy as np
from pyvista import examples
import sys
import json
import requests

def download_world_borders():
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    print(f"正在从 {url} 下载全球国界线数据...")
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        all_pts = []
        border_cells = []
        country_ids_borders = []
        country_names = {}
        country_geoms = {}
        curr_pt_idx = 0

        for idx, feature in enumerate(data['features']):
            name = feature['properties'].get('name', f"Country {idx}")
            country_names[idx] = name
            country_geoms[idx] = []

            geom = feature['geometry']
            coords = geom['coordinates']
            polys = [coords] if geom['type'] == 'Polygon' else coords

            for poly in polys:
                rings_for_country = []
                for ring in poly:
                    if len(ring) < 3:
                        continue
                    ring_arr = np.asarray(ring, dtype=float)
                    rings_for_country.append(ring_arr)

                    ring_pts = []
                    for lon, lat in ring:
                        phi = np.deg2rad(lat)
                        theta = np.deg2rad(lon)
                        x = np.cos(phi) * np.cos(theta)
                        y = np.cos(phi) * np.sin(theta)
                        z = np.sin(phi)
                        ring_pts.append([x, y, z])

                    num_pts = len(ring_pts)
                    if num_pts < 2:
                        continue

                    all_pts.extend(ring_pts)
                    line_cell = [num_pts] + list(range(curr_pt_idx, curr_pt_idx + num_pts))
                    border_cells.append(line_cell)
                    country_ids_borders.append(idx)
                    curr_pt_idx += num_pts

                if rings_for_country:
                    outer = rings_for_country[0]
                    holes = rings_for_country[1:]
                    min_lon = float(np.min(outer[:, 0]))
                    max_lon = float(np.max(outer[:, 0]))
                    min_lat = float(np.min(outer[:, 1]))
                    max_lat = float(np.max(outer[:, 1]))
                    country_geoms[idx].append(
                        {
                            "outer": outer,
                            "holes": holes,
                            "bbox": (min_lon, max_lon, min_lat, max_lat),
                            "crosses_dateline": (max_lon - min_lon) > 180.0,
                        }
                    )

        if not all_pts:
            return None, {}, {}

        pts_array = np.array(all_pts)
        borders = pv.PolyData(pts_array, lines=np.hstack(border_cells))
        borders.cell_data['country_id'] = np.array(country_ids_borders)
        return borders, country_names, country_geoms
    except Exception as e:
        print(f"加载国界线数据失败: {e}")
        return None, {}, {}


def _point_in_ring(lon, lat, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _normalize_lon(lon):
    v = ((lon + 180.0) % 360.0) - 180.0
    return 180.0 if v == -180.0 else v


def _point_in_country(lon, lat, polygons):
    lon = _normalize_lon(lon)
    for poly in polygons:
        min_lon, max_lon, min_lat, max_lat = poly["bbox"]
        test_lon = lon
        outer = poly["outer"]
        holes = poly["holes"]

        if poly["crosses_dateline"]:
            if test_lon < 0:
                test_lon += 360.0
            outer_test = outer.copy()
            outer_test[:, 0] = np.where(outer_test[:, 0] < 0, outer_test[:, 0] + 360.0, outer_test[:, 0])
            if not (np.min(outer_test[:, 0]) <= test_lon <= np.max(outer_test[:, 0]) and min_lat <= lat <= max_lat):
                continue
            if not _point_in_ring(test_lon, lat, outer_test):
                continue
            in_hole = False
            for hole in holes:
                hole_test = hole.copy()
                hole_test[:, 0] = np.where(hole_test[:, 0] < 0, hole_test[:, 0] + 360.0, hole_test[:, 0])
                if _point_in_ring(test_lon, lat, hole_test):
                    in_hole = True
                    break
            if not in_hole:
                return True
        else:
            if not (min_lon <= test_lon <= max_lon and min_lat <= lat <= max_lat):
                continue
            if not _point_in_ring(test_lon, lat, outer):
                continue
            in_hole = False
            for hole in holes:
                if _point_in_ring(test_lon, lat, hole):
                    in_hole = True
                    break
            if not in_hole:
                return True
    return False

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
        
    borders, country_names, country_geoms = download_world_borders()

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
    plotter = pv.Plotter(title="Python 3D 虚拟地球仪 - 国家高亮交互")
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

    # 5.5 高亮图层
    highlight_actor = None
    plotter.add_text("", position='upper_left', font_size=12, color='yellow', name="country_label")
    
    # 6. 滑杆回调函数：调节地形夸张比例
    def update_exaggeration(value):
        # value 范围 1.0 到 5.0
        base_factor = 40.0
        current_factor = value * base_factor
        
        displacement = (scalars / 6371000.0) * current_factor
        topo.points = pts_orig + unit_vectors * displacement[:, np.newaxis]
        
        # 轮廓线和国界线同步浮动
        height_offset = 1.0 + (value * 0.005)
        if coastlines and coast_actor:
            coastlines.points = coast_pts_orig * height_offset
        if borders and borders_actor:
            borders.points = borders_pts_orig * (height_offset + 0.001)
            
        plotter.render()

    # 6.5 国家悬浮高亮逻辑
    state = {'active_id': -1}
    
    # 使用 CellPicker 以获取面索引
    import vtk
    cell_picker = vtk.vtkCellPicker()
    cell_picker.SetTolerance(0.005) # 设置拾取容差
    
    def on_mouse_move(_obj, _event):
        nonlocal highlight_actor
        click_pos = plotter.iren.get_event_position()
        cell_picker.Pick(click_pos[0], click_pos[1], 0, plotter.renderer)
        picked_actor = cell_picker.GetActor()
        cid = -1
        cell_id = cell_picker.GetCellId()
        pick_pos = None
        if picked_actor is not None and cell_id != -1:
            pick_pos = np.array(cell_picker.GetPickPosition(), dtype=float)
            norm = np.linalg.norm(pick_pos)
            if norm > 0:
                dir_vec = pick_pos / norm
                lon = np.rad2deg(np.arctan2(dir_vec[1], dir_vec[0]))
                lat = np.rad2deg(np.arcsin(np.clip(dir_vec[2], -1.0, 1.0)))
                for country_id, polygons in country_geoms.items():
                    if _point_in_country(lon, lat, polygons):
                        cid = country_id
                        break
                if cid == -1 and borders is not None:
                    nearest_cell = borders.find_closest_cell(pick_pos)
                    if nearest_cell is not None and nearest_cell >= 0:
                        cid = int(borders.cell_data['country_id'][nearest_cell])
        
        if cid != state['active_id']:
            state['active_id'] = cid
            if cid != -1:
                name = country_names.get(cid, "Unknown")
                # 更新文本
                plotter.add_text(f"当前国家: {name}", position='upper_left', font_size=12, color='yellow', name="country_label")
                
                mask = borders.cell_data['country_id'] == cid
                selected_border = borders.extract_cells(mask)
                
                if highlight_actor:
                    plotter.remove_actor(highlight_actor)
                highlight_actor = plotter.add_mesh(selected_border, color="yellow", line_width=4, name="highlight", pickable=False)
            else:
                plotter.add_text("", position='upper_left', font_size=12, color='yellow', name="country_label")
                if highlight_actor:
                    plotter.remove_actor(highlight_actor)
                    highlight_actor = None
            plotter.render()

    # 绑定鼠标移动事件
    plotter.iren.add_observer("MouseMoveEvent", on_mouse_move)

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
