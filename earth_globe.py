import pyvista as pv
import numpy as np
from pyvista import examples
import sys

def create_virtual_globe():
    print("正在加载地球地形数据和轮廓...")
    
    # 1. 加载地形网格 (带有 'altitude' 标量数据)
    # 该数据集包含了全球的海拔/深度信息
    try:
        topo = examples.download_topo_global()
    except Exception as e:
        print(f"下载地形数据失败: {e}")
        topo = pv.Sphere(radius=1.0, theta_resolution=120, phi_resolution=120)
        topo.point_data['altitude'] = np.zeros(topo.n_points)

    # 2. 加载海岸线（大洲轮廓）
    try:
        coastlines = examples.download_coastlines()
    except Exception as e:
        print(f"加载海岸线失败: {e}")
        coastlines = None

    # 3. 增强地形效果 (位移顶点实现 3D 地貌)
    print("正在生成 3D 地貌位移...")
    scalars = topo.point_data['altitude']
    
    # 夸大倍数：为了视觉效果，让山脉清晰可见
    # 地球半径设为 1.0，海拔位移根据比例缩放
    exaggeration = 40.0 
    displacement = (scalars / 6371000.0) * exaggeration
    
    pts = topo.points
    # 获取指向球心外的单位向量
    norms = np.linalg.norm(pts, axis=1)
    norms[norms == 0] = 1.0
    unit_vectors = pts / norms[:, np.newaxis]
    
    # 应用位移：根据海拔高度拉伸顶点
    topo.points = pts + unit_vectors * displacement[:, np.newaxis]

    # 如果有海岸线，也需要略微抬高以防止与地表重叠
    if coastlines:
        # 海岸线原本是在半径为 1.0 的球面上，我们将其统一抬高到地貌之上
        coastlines.points *= 1.01

    # 4. 设置可视化界面
    # 彻底解决重绘残留（Ghosting/Not erasing）的核心策略：
    # 1. 禁用多重采样（硬件 MSAA 在某些 GPU 上会导致深度缓冲清除失败）
    # 2. 移除背景图渲染器（Background Image Renderer），因为它是残留的最常见来源
    # 3. 使用 SSAA（超采样抗锯齿）替代 FXAA，SSAA 会强制刷新整个渲染表面
    pv.global_theme.multi_samples = 0
    plotter = pv.Plotter(
        title="Python 3D 虚拟地形仪 (纯地貌+轮廓模式)",
        lighting='light_kit'
    )
    plotter.enable_anti_aliasing('ssaa') # 使用更彻底的 SSAA 抗锯齿
    plotter.set_background("black") # 纯黑背景，确保擦除干净

    # 添加地形网格：使用 'terrain' 色带根据海拔着色
    earth_mesh = plotter.add_mesh(
        topo, 
        scalars='altitude', 
        cmap="terrain", 
        smooth_shading=True, 
        name="earth",
        clim=[-8000, 8848],
        show_scalar_bar=True,
        scalar_bar_args={'title': '海拔 (m)', 'n_labels': 5}
    )

    # 添加大洲轮廓线
    coast_mesh = None
    if coastlines:
        coast_mesh = plotter.add_mesh(
            coastlines, 
            color="white", 
            line_width=2, 
            label="Coastlines", 
            name="coastlines"
        )

    # 5. 移除星空背景图（它是产生重绘残留的主要原因）
    # 如果您需要星空，我们可以稍后用点云（Points）来实现

    # 6. 交互逻辑与说明
    def toggle_coastlines():
        if coast_mesh:
            visible = not coast_mesh.GetVisibility()
            coast_mesh.SetVisibility(visible)
            print(f"轮廓线显示: {'开' if visible else '关'}")

    def reset_view():
        plotter.camera_position = [(3.5, 3.5, 3.5), (0, 0, 0), (0, 0, 1)]
        print("视角已重置")

    # 绑定键盘按键
    plotter.add_key_event("c", toggle_coastlines)
    plotter.add_key_event("r", reset_view)

    print("\n" + "="*30)
    print("      虚拟地球仪已启动")
    print("="*30)
    print("操作说明:")
    print("- [鼠标左键]   : 自由旋转 (游历地貌)")
    print("- [鼠标右键]   : 缩放")
    print("- [键盘 'C']   : 开关大洲轮廓线")
    print("- [键盘 'R']   : 重置初始视角")
    print("- [键盘 'Q']   : 退出程序")
    print("="*30)

    # 初始视角
    plotter.camera_position = [(3.5, 3.5, 3.5), (0, 0, 0), (0, 0, 1)]
    
    # 启动显示
    plotter.show()

if __name__ == "__main__":
    create_virtual_globe()
