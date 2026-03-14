import pyvista as pv
import numpy as np
from pyvista import examples
import sys

def create_virtual_globe():
    print("正在加载地球地形数据和轮廓...")
    
    # 1. 加载地形网格
    try:
        topo = examples.download_topo_global()
    except Exception as e:
        print(f"下载地形数据失败: {e}")
        topo = pv.Sphere(radius=1.0, theta_resolution=120, phi_resolution=120)
        topo.point_data['altitude'] = np.zeros(topo.n_points)

    # 2. 加载地表贴图和海岸线
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
    
    # 保存海岸线原始位置
    if coastlines:
        coast_pts_orig = coastlines.points.copy()

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
        coast_actor = plotter.add_mesh(coastlines, color="white", line_width=2, name="coastlines")

    # 6. 滑杆回调函数：调节地形夸张比例
    def update_exaggeration(value):
        # value 范围 1.0 到 5.0
        # 1:1 比例下，真实位移 = 海拔 / 6371000.0
        # 基础夸大倍数（为了肉眼可见，1.0 对应 40 倍，5.0 对应 200 倍）
        base_factor = 40.0
        current_factor = value * base_factor
        
        displacement = (scalars / 6371000.0) * current_factor
        topo.points = pts_orig + unit_vectors * displacement[:, np.newaxis]
        
        # 海岸线同步浮动，保持在地形上方一点点
        if coastlines and coast_actor:
            coastlines.points = coast_pts_orig * (1.0 + (value * 0.005))
            
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

    def reset_view():
        plotter.camera_position = [(3.5, 3.5, 3.5), (0, 0, 0), (0, 0, 1)]

    plotter.add_key_event("c", toggle_coastlines)
    plotter.add_key_event("r", reset_view)

    print("\n" + "="*30)
    print("      虚拟地球仪已启动")
    print("="*30)
    print("操作说明:")
    print("- [滑杆]       : 调节地形夸张比例 (1:1 ~ 1:5)")
    print("- [鼠标左键]   : 自由旋转")
    print("- [键盘 'C']   : 开关大洲轮廓线")
    print("- [键盘 'R']   : 重置初始视角")
    print("- [键盘 'Q']   : 退出程序")
    print("="*30)

    plotter.camera_position = [(3.5, 3.5, 3.5), (0, 0, 0), (0, 0, 1)]
    plotter.show()

if __name__ == "__main__":
    create_virtual_globe()
