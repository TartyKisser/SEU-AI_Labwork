import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import matplotlib

matplotlib.use('TkAgg')
import random
from scipy.spatial import distance


class Obstacle:
    """障碍物类"""

    def __init__(self, x, y, width, height, shape='rectangle'):
        """初始化障碍物

        参数:
            x, y: 障碍物左下角坐标（矩形）或中心坐标（圆形）
            width: 宽度（矩形）或半径（圆形）
            height: 高度（矩形，圆形时忽略）
            shape: 形状类型 ('rectangle' 或 'circle')
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.shape = shape
        self.color = 'gray'

    def contains_point(self, point):
        """检查点是否在障碍物内部"""
        px, py = point

        if self.shape == 'rectangle':
            return (self.x <= px <= self.x + self.width and
                    self.y <= py <= self.y + self.height)
        elif self.shape == 'circle':
            # 对于圆形，x,y是中心坐标，width是半径
            center_distance = np.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
            return center_distance <= self.width

        return False

    def distance_to_point(self, point):
        """计算点到障碍物边界的最短距离"""
        px, py = point

        if self.shape == 'rectangle':
            # 计算点到矩形的最短距离
            dx = max(self.x - px, 0, px - (self.x + self.width))
            dy = max(self.y - py, 0, py - (self.y + self.height))
            return np.sqrt(dx * dx + dy * dy)

        elif self.shape == 'circle':
            # 计算点到圆的最短距离
            center_distance = np.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
            return max(0, center_distance - self.width)

        return float('inf')

    def get_repulsion_force(self, point, max_distance=15.0):
        """计算障碍物对点的排斥力"""
        distance = self.distance_to_point(point)

        if distance >= max_distance:
            return np.array([0.0, 0.0])

        # 计算从障碍物指向点的方向
        px, py = point

        if self.shape == 'rectangle':
            # 找到矩形上距离点最近的点
            closest_x = np.clip(px, self.x, self.x + self.width)
            closest_y = np.clip(py, self.y, self.y + self.height)

            direction = np.array([px - closest_x, py - closest_y])

        elif self.shape == 'circle':
            # 从圆心指向点的方向
            direction = np.array([px - self.x, py - self.y])

        # 归一化方向向量
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            # 如果点在障碍物中心，随机选择一个方向
            direction = np.random.randn(2)
            direction_norm = np.linalg.norm(direction)

        if direction_norm > 0:
            direction = direction / direction_norm

            # 排斥力强度与距离成反比
            if distance == 0:
                strength = 1.0  # 最大排斥力
            else:
                strength = max(0, (max_distance - distance) / max_distance)

            return direction * strength

        return np.array([0.0, 0.0])

    def check_line_intersection(self, start_point, end_point):
        """检查线段是否与障碍物相交"""
        if self.shape == 'rectangle':
            return self._line_intersects_rectangle(start_point, end_point)
        elif self.shape == 'circle':
            return self._line_intersects_circle(start_point, end_point)
        return False

    def _line_intersects_rectangle(self, start_point, end_point):
        """检查线段是否与矩形相交"""
        x1, y1 = start_point
        x2, y2 = end_point

        # 矩形的四条边
        rect_lines = [
            ((self.x, self.y), (self.x + self.width, self.y)),  # 底边
            ((self.x + self.width, self.y), (self.x + self.width, self.y + self.height)),  # 右边
            ((self.x + self.width, self.y + self.height), (self.x, self.y + self.height)),  # 顶边
            ((self.x, self.y + self.height), (self.x, self.y))  # 左边
        ]

        for (rx1, ry1), (rx2, ry2) in rect_lines:
            if self._line_segments_intersect((x1, y1), (x2, y2), (rx1, ry1), (rx2, ry2)):
                return True

        return False

    def _line_intersects_circle(self, start_point, end_point):
        """检查线段是否与圆相交"""
        x1, y1 = start_point
        x2, y2 = end_point
        cx, cy = self.x, self.y
        r = self.width

        # 计算线段到圆心的最短距离
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            # 线段退化为点
            return np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2) <= r

        t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        distance = np.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)
        return distance <= r

    def _line_segments_intersect(self, p1, p2, p3, p4):
        """检查两个线段是否相交"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False  # 平行线

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        return 0 <= t <= 1 and 0 <= u <= 1

    def draw(self, ax):
        """在matplotlib轴上绘制障碍物"""
        if self.shape == 'rectangle':
            rect = patches.Rectangle(
                (self.x, self.y), self.width, self.height,
                linewidth=2, edgecolor='black', facecolor=self.color, alpha=0.7
            )
            ax.add_patch(rect)
        elif self.shape == 'circle':
            circle = plt.Circle(
                (self.x, self.y), self.width,
                linewidth=2, edgecolor='black', facecolor=self.color, alpha=0.7
            )
            ax.add_patch(circle)


class MapExplorer:
    """地图探索器，用于追踪者的地图遍历功能"""

    def __init__(self, boundary, grid_size=20, obstacles=None):
        """初始化地图探索器

        参数:
            boundary: 边界 [min_x, max_x, min_y, max_y]
            grid_size: 网格大小，用于将地图分割成小区域
            obstacles: 障碍物列表
        """
        self.boundary = boundary
        self.grid_size = grid_size
        self.obstacles = obstacles or []

        min_x, max_x, min_y, max_y = boundary
        self.grid_width = int((max_x - min_x) / grid_size)
        self.grid_height = int((max_y - min_y) / grid_size)

        # 创建探索状态网格，False表示未探索，True表示已探索
        self.explored_grid = np.zeros((self.grid_width, self.grid_height), dtype=bool)

        # 为每个追踪者记录目标网格点
        self.pursuer_targets = {}

        # 为每个追踪者记录被阻挡的目标点列表
        self.blocked_targets = {}

    def get_grid_position(self, position):
        """将世界坐标转换为网格坐标"""
        min_x, max_x, min_y, max_y = self.boundary
        grid_x = int((position[0] - min_x) / self.grid_size)
        grid_y = int((position[1] - min_y) / self.grid_size)

        # 确保网格坐标在有效范围内
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_y = max(0, min(self.grid_height - 1, grid_y))

        return grid_x, grid_y

    def get_world_position(self, grid_x, grid_y):
        """将网格坐标转换为世界坐标（网格中心点）"""
        min_x, max_x, min_y, max_y = self.boundary
        world_x = min_x + (grid_x + 0.5) * self.grid_size
        world_y = min_y + (grid_y + 0.5) * self.grid_size
        return np.array([world_x, world_y])

    def mark_explored(self, position, sensor_range):
        """标记以position为中心，sensor_range为半径的区域为已探索"""
        center_grid_x, center_grid_y = self.get_grid_position(position)

        # 计算需要标记的网格范围
        grid_radius = int(sensor_range / self.grid_size) + 1

        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                grid_x = center_grid_x + dx
                grid_y = center_grid_y + dy

                if (0 <= grid_x < self.grid_width and
                        0 <= grid_y < self.grid_height):

                    # 检查网格中心是否在传感器范围内
                    grid_world_pos = self.get_world_position(grid_x, grid_y)
                    distance = np.linalg.norm(position - grid_world_pos)

                    if distance <= sensor_range:
                        self.explored_grid[grid_x, grid_y] = True

    def is_path_blocked(self, start_position, target_position):
        """检查从起始位置到目标位置的路径是否被障碍物阻挡"""
        for obstacle in self.obstacles:
            if obstacle.check_line_intersection(start_position, target_position):
                return True
        return False

    def find_alternative_target(self, pursuer_id, current_position, blocked_target):
        """当目标被阻挡时，寻找替代目标"""
        # 初始化被阻挡目标列表
        if pursuer_id not in self.blocked_targets:
            self.blocked_targets[pursuer_id] = set()

        # 将被阻挡的目标添加到列表中（转换为元组以便存储在set中）
        blocked_target_tuple = tuple(blocked_target)
        self.blocked_targets[pursuer_id].add(blocked_target_tuple)

        # 寻找未被阻挡的替代目标
        best_target = None
        min_distance = float('inf')

        # 搜索范围：先搜索附近的网格，再逐渐扩大范围
        max_search_radius = max(self.grid_width, self.grid_height)

        for search_radius in range(1, max_search_radius):
            candidates_found = False

            for grid_x in range(self.grid_width):
                for grid_y in range(self.grid_height):
                    # 跳过已探索的网格
                    if self.explored_grid[grid_x, grid_y]:
                        continue

                    candidate_pos = self.get_world_position(grid_x, grid_y)
                    candidate_tuple = tuple(candidate_pos)

                    # 跳过已知被阻挡的目标
                    if candidate_tuple in self.blocked_targets[pursuer_id]:
                        continue

                    # 检查是否在当前搜索半径内
                    distance = np.linalg.norm(current_position - candidate_pos)
                    grid_distance = max(abs(grid_x - self.get_grid_position(current_position)[0]),
                                        abs(grid_y - self.get_grid_position(current_position)[1]))

                    if grid_distance != search_radius:
                        continue

                    # 检查路径是否被阻挡
                    if self.is_path_blocked(current_position, candidate_pos):
                        # 将这个目标也添加到被阻挡列表中
                        self.blocked_targets[pursuer_id].add(candidate_tuple)
                        continue

                    candidates_found = True

                    if distance < min_distance:
                        min_distance = distance
                        best_target = candidate_pos

            # 如果在当前搜索半径内找到了候选目标，就不需要继续扩大搜索范围
            if candidates_found and best_target is not None:
                break

        return best_target

    def get_unexplored_target(self, pursuer_id, current_position):
        """为指定追踪者获取下一个未探索的目标点"""
        # 如果已有目标且目标还未到达
        if pursuer_id in self.pursuer_targets:
            target_pos = self.pursuer_targets[pursuer_id]
            distance_to_target = np.linalg.norm(current_position - target_pos)

            # 如果距离目标很近，认为已到达，需要选择新目标
            if distance_to_target > self.grid_size * 0.5:
                # 检查当前路径是否被阻挡
                if self.is_path_blocked(current_position, target_pos):
                    # 路径被阻挡，寻找替代目标
                    print(f"追踪者 {pursuer_id} 的路径被阻挡，正在寻找替代目标...")
                    alternative_target = self.find_alternative_target(pursuer_id, current_position, target_pos)

                    if alternative_target is not None:
                        self.pursuer_targets[pursuer_id] = alternative_target
                        return alternative_target
                    else:
                        # 如果找不到替代目标，清除当前目标，重新搜索
                        del self.pursuer_targets[pursuer_id]
                else:
                    # 路径没有被阻挡，继续前往当前目标
                    return target_pos

        # 寻找最近的未探索网格点
        best_target = None
        min_distance = float('inf')

        # 初始化被阻挡目标列表
        if pursuer_id not in self.blocked_targets:
            self.blocked_targets[pursuer_id] = set()

        for grid_x in range(self.grid_width):
            for grid_y in range(self.grid_height):
                if not self.explored_grid[grid_x, grid_y]:
                    candidate_pos = self.get_world_position(grid_x, grid_y)
                    candidate_tuple = tuple(candidate_pos)

                    # 跳过已知被阻挡的目标
                    if candidate_tuple in self.blocked_targets[pursuer_id]:
                        continue

                    distance = np.linalg.norm(current_position - candidate_pos)

                    # 检查路径是否被阻挡
                    if self.is_path_blocked(current_position, candidate_pos):
                        # 将被阻挡的目标添加到列表中
                        self.blocked_targets[pursuer_id].add(candidate_tuple)
                        continue

                    if distance < min_distance:
                        min_distance = distance
                        best_target = candidate_pos

        # 更新追踪者的目标
        if best_target is not None:
            self.pursuer_targets[pursuer_id] = best_target

        return best_target

    def is_fully_explored(self):
        """检查地图是否已完全探索"""
        return np.all(self.explored_grid)

    def get_exploration_progress(self):
        """获取探索进度（0-1）"""
        total_grids = self.grid_width * self.grid_height
        explored_grids = np.sum(self.explored_grid)
        return explored_grids / total_grids if total_grids > 0 else 0

    def reset(self):
        """重置探索状态"""
        self.explored_grid.fill(False)
        self.pursuer_targets.clear()
        self.blocked_targets.clear()


class Robot:
    """基础机器人类，用于表示追踪者和逃避者的共有特性"""

    def __init__(self, x, y, speed, radius, color, boundary, id, obstacles=None):
        """初始化机器人"""
        self.position = np.array([x, y], dtype=float)
        self.speed = speed
        self.radius = radius
        self.color = color
        self.boundary = boundary
        self.obstacles = obstacles or []
        self.history = [self.position.copy()]  # 存储历史轨迹
        self.max_history = 50  # 记录的最大历史位置点数
        self.id = id  # 唯一标识

    def is_position_valid(self, position):
        """检查位置是否有效（不与障碍物碰撞）"""
        # 检查边界
        min_x, max_x, min_y, max_y = self.boundary
        if (position[0] - self.radius < min_x or position[0] + self.radius > max_x or
                position[1] - self.radius < min_y or position[1] + self.radius > max_y):
            return False

        # 检查与障碍物的碰撞
        for obstacle in self.obstacles:
            if obstacle.distance_to_point(position) < self.radius:
                return False

        return True

    def get_obstacle_avoidance_force(self):
        """计算避障力"""
        total_force = np.array([0.0, 0.0])

        for obstacle in self.obstacles:
            repulsion_force = obstacle.get_repulsion_force(self.position, max_distance=20.0)
            total_force += repulsion_force

        return total_force

    def move(self, direction):
        """按指定方向移动机器人"""
        # 添加避障力
        obstacle_force = self.get_obstacle_avoidance_force()
        combined_direction = direction + obstacle_force * 2.0  # 增加避障权重

        # 归一化方向向量
        direction_norm = np.linalg.norm(combined_direction)
        if direction_norm > 0:
            combined_direction = combined_direction / direction_norm

        # 按速度和方向计算新位置
        new_position = self.position + combined_direction * self.speed

        # 边界检查
        min_x, max_x, min_y, max_y = self.boundary
        new_position[0] = max(min_x + self.radius, min(max_x - self.radius, new_position[0]))
        new_position[1] = max(min_y + self.radius, min(max_y - self.radius, new_position[1]))

        # 检查新位置是否与障碍物碰撞
        if self.is_position_valid(new_position):
            self.position = new_position
        else:
            # 如果碰撞，尝试沿着边界滑动
            self.position = self.find_sliding_position(new_position)

        # 更新历史轨迹
        self.history.append(self.position.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def find_sliding_position(self, target_position):
        """当目标位置无效时，寻找一个有效的滑动位置"""
        # 尝试多个方向来避开障碍物
        angles = np.linspace(0, 2 * np.pi, 8)

        for angle in angles:
            slide_direction = np.array([np.cos(angle), np.sin(angle)])
            test_position = self.position + slide_direction * self.speed

            # 边界检查
            min_x, max_x, min_y, max_y = self.boundary
            test_position[0] = max(min_x + self.radius, min(max_x - self.radius, test_position[0]))
            test_position[1] = max(min_y + self.radius, min(max_y - self.radius, test_position[1]))

            if self.is_position_valid(test_position):
                return test_position

        # 如果所有方向都被阻挡，保持当前位置
        return self.position.copy()

    def distance_to(self, other):
        """计算到另一个机器人的距离"""
        return np.linalg.norm(self.position - other.position)

    def is_captured(self, pursuer, capture_distance):
        """检查是否被追捕者捕获"""
        return self.distance_to(pursuer) < capture_distance

    def can_see(self, target):
        """检查是否能看到目标（视线是否被障碍物阻挡）"""
        for obstacle in self.obstacles:
            if obstacle.check_line_intersection(self.position, target.position):
                return False
        return True

    def draw(self, ax):
        """在matplotlib轴上绘制机器人"""
        circle = plt.Circle(self.position, self.radius, color=self.color)
        ax.add_artist(circle)

        # 绘制历史轨迹
        if len(self.history) > 1:
            history_array = np.array(self.history)
            ax.plot(history_array[:, 0], history_array[:, 1], color=self.color, alpha=0.3, linewidth=1)


class SharedInformation:
    """信息共享类，用于机器人之间的信息交换"""

    def __init__(self):
        # 追踪者共享的逃避者信息 {evader_id: {position, last_seen, detected_by}}
        self.pursuers_knowledge = {}

        # 逃避者共享的追踪者信息 {pursuer_id: {position, last_seen, detected_by}}
        self.evaders_knowledge = {}

        # 信息记忆时长（步数）
        self.memory_duration = 20

    def update_evader_info(self, evader, detecting_pursuer, current_step):
        """更新追踪者关于逃避者的共享信息"""
        self.pursuers_knowledge[evader.id] = {
            'position': evader.position.copy(),
            'last_seen': current_step,
            'detected_by': detecting_pursuer.id
        }

    def update_pursuer_info(self, pursuer, detecting_evader, current_step):
        """更新逃避者关于追踪者的共享信息"""
        self.evaders_knowledge[pursuer.id] = {
            'position': pursuer.position.copy(),
            'last_seen': current_step,
            'detected_by': detecting_evader.id
        }

    def clean_old_information(self, current_step):
        """清除过期的信息"""
        # 清除过期的追踪者知识
        self.pursuers_knowledge = {
            evader_id: info for evader_id, info in self.pursuers_knowledge.items()
            if current_step - info['last_seen'] <= self.memory_duration
        }

        # 清除过期的逃避者知识
        self.evaders_knowledge = {
            pursuer_id: info for pursuer_id, info in self.evaders_knowledge.items()
            if current_step - info['last_seen'] <= self.memory_duration
        }

    def remove_evader_info(self, evader_id):
        """从共享信息中移除指定逃避者的信息"""
        if evader_id in self.pursuers_knowledge:
            del self.pursuers_knowledge[evader_id]


class Pursuer(Robot):
    """追踪者类，继承自Robot"""

    def __init__(self, x, y, speed, radius, strategy, sensor_range, boundary, id, shared_info, obstacles=None,
                 map_explorer=None):
        """初始化追踪者"""
        super().__init__(x, y, speed, radius, 'red', boundary, id, obstacles)
        self.strategy = strategy
        self.sensor_range = sensor_range
        self.target = None
        self.target_id = None  # 目标ID
        self.shared_info = shared_info
        self.map_explorer = map_explorer
        self.communication_range = sensor_range * 2  # 通信范围比感知范围大
        self.exploration_mode = False  # 是否处于地图探索模式
        self.exploration_target = None  # 探索目标点

    def detect_evaders(self, evaders, current_step):
        """检测感知范围内的逃避者并更新共享信息"""
        detected = []
        for evader in evaders:
            distance = self.distance_to(evader)
            # 检查距离和视线（是否被障碍物阻挡）
            if distance <= self.sensor_range and self.can_see(evader):
                # 发现了一个逃避者，更新共享信息
                self.shared_info.update_evader_info(evader, self, current_step)
                detected.append(evader)

        return detected

    def choose_target(self, evaders, current_step):
        """从逃避者中选择目标"""
        if not evaders:
            return None, None

        # 直接检测的逃避者
        directly_detected = self.detect_evaders(evaders, current_step)

        # 收集所有可用的目标信息（直接检测 + 共享信息）
        potential_targets = {}

        # 添加直接检测的目标
        for evader in directly_detected:
            potential_targets[evader.id] = {
                'evader': evader,
                'position': evader.position,
                'distance': self.distance_to(evader),
                'directly_detected': True
            }

        # 添加共享信息中的目标
        for evader_id, info in self.shared_info.pursuers_knowledge.items():
            # 如果这个逃避者已经被直接检测到，跳过
            if evader_id in potential_targets:
                continue

            # 计算到共享位置的距离
            shared_position = info['position']
            distance_to_shared = np.linalg.norm(self.position - shared_position)

            # 查找对应的逃避者对象
            target_evader = next((e for e in evaders if e.id == evader_id), None)

            if target_evader:
                potential_targets[evader_id] = {
                    'evader': target_evader,
                    'position': shared_position,
                    'distance': distance_to_shared,
                    'directly_detected': False,
                    'info_age': current_step - info['last_seen']
                }

        # 如果没有潜在目标，返回None
        if not potential_targets:
            return None, None

        # 策略选择
        target_id = None
        target_evader = None

        if self.strategy == 'nearest':
            # 选择最近的目标
            nearest_id = min(potential_targets.keys(),
                             key=lambda k: potential_targets[k]['distance'])
            target_id = nearest_id
            target_evader = potential_targets[nearest_id]['evader']

        elif self.strategy == 'random':
            # 随机选择一个目标
            target_id = random.choice(list(potential_targets.keys()))
            target_evader = potential_targets[target_id]['evader']

        return target_evader, target_id

    def get_exploration_direction(self):
        """获取地图探索方向"""
        if self.map_explorer is None:
            # 如果没有地图探索器，使用随机方向
            return np.random.randn(2)

        # 获取下一个探索目标
        self.exploration_target = self.map_explorer.get_unexplored_target(
            self.id, self.position)

        if self.exploration_target is not None:
            # 朝向探索目标移动
            direction = self.exploration_target - self.position
            return direction
        else:
            # 如果地图已完全探索，使用随机搜索
            return np.random.randn(2)

    def update(self, evaders, other_pursuers, current_step):
        """更新追踪者状态"""
        # 更新地图探索状态
        if self.map_explorer is not None:
            self.map_explorer.mark_explored(self.position, self.sensor_range)

        # 选择目标
        target_evader, target_id = self.choose_target(evaders, current_step)
        self.target = target_evader
        self.target_id = target_id

        # 确定移动方向
        if self.target:
            # 发现目标，退出探索模式
            self.exploration_mode = False

            # 使用共享位置或者当前感知的位置
            target_position = self.target.position

            # 向目标移动
            direction = target_position - self.position
        else:
            # 没有目标时，检查是否所有追踪者都没有发现目标
            any_pursuer_has_target = any(p.target is not None for p in [self] + other_pursuers)

            if not any_pursuer_has_target:
                # 所有追踪者都没有目标，进入地图探索模式
                self.exploration_mode = True
                direction = self.get_exploration_direction()
            else:
                # 有其他追踪者发现了目标，协助搜索
                self.exploration_mode = False
                # 可以朝向最近的已知目标位置移动，或进行局部搜索
                if self.shared_info.pursuers_knowledge:
                    # 朝向最近的共享目标位置
                    nearest_shared_pos = None
                    min_distance = float('inf')

                    for evader_id, info in self.shared_info.pursuers_knowledge.items():
                        distance = np.linalg.norm(self.position - info['position'])
                        if distance < min_distance:
                            min_distance = distance
                            nearest_shared_pos = info['position']

                    if nearest_shared_pos is not None:
                        direction = nearest_shared_pos - self.position
                    else:
                        direction = np.random.randn(2)
                else:
                    direction = np.random.randn(2)

        self.move(direction)


class Evader(Robot):
    """逃避者类，继承自Robot"""

    def __init__(self, x, y, speed, radius, strategy, sensor_range, boundary, id, shared_info, obstacles=None):
        """初始化逃避者"""
        super().__init__(x, y, speed, radius, 'blue', boundary, id, obstacles)
        self.strategy = strategy
        self.sensor_range = sensor_range
        self.shared_info = shared_info
        self.communication_range = sensor_range * 2  # 通信范围

    def detect_pursuers(self, pursuers, current_step):
        """检测感知范围内的追踪者并更新共享信息"""
        detected = []
        for pursuer in pursuers:
            distance = self.distance_to(pursuer)
            # 检查距离和视线（是否被障碍物阻挡）
            if distance <= self.sensor_range and self.can_see(pursuer):
                # 发现一个追踪者，更新共享信息
                self.shared_info.update_pursuer_info(pursuer, self, current_step)
                detected.append(pursuer)

        return detected

    def get_threats(self, pursuers, current_step):
        """获取所有潜在威胁（直接感知 + 共享信息）"""
        # 直接检测的追踪者
        directly_detected = self.detect_pursuers(pursuers, current_step)

        # 收集所有威胁信息
        threats = []

        # 添加直接检测的威胁
        for pursuer in directly_detected:
            threats.append({
                'pursuer': pursuer,
                'position': pursuer.position,
                'distance': self.distance_to(pursuer),
                'directly_detected': True
            })

        # 添加共享信息中的威胁
        for pursuer_id, info in self.shared_info.evaders_knowledge.items():
            # 检查是否已经直接检测到
            if any(p.id == pursuer_id for p in directly_detected):
                continue

            # 计算到共享位置的距离
            shared_position = info['position']
            distance_to_shared = np.linalg.norm(self.position - shared_position)

            # 查找对应的追踪者对象
            threat_pursuer = next((p for p in pursuers if p.id == pursuer_id), None)

            if threat_pursuer:
                threats.append({
                    'pursuer': threat_pursuer,
                    'position': shared_position,
                    'distance': distance_to_shared,
                    'directly_detected': False,
                    'info_age': current_step - info['last_seen']
                })

        return threats

    def update(self, pursuers, other_evaders, current_step):
        """更新逃避者状态"""
        # 获取威胁
        threats = self.get_threats(pursuers, current_step)

        if threats:
            if self.strategy == 'avoid_nearest':
                # 避开最近的追踪者
                nearest_threat = min(threats, key=lambda t: t['distance'])
                direction = self.position - nearest_threat['position']

            elif self.strategy == 'avoid_all':
                # 考虑所有威胁，综合计算逃离方向
                direction = np.zeros(2)
                for threat in threats:
                    # 从每个威胁逃离，距离越近贡献越大
                    flee_direction = self.position - threat['position']
                    flee_direction_norm = np.linalg.norm(flee_direction)
                    if flee_direction_norm > 0:
                        # 权重与距离成反比
                        weight = 1.0 / max(0.1, flee_direction_norm)  # 避免除以零
                        direction += weight * flee_direction

                        # 直接感知的威胁权重更高
                        if threat.get('directly_detected', False):
                            direction += 0.5 * weight * flee_direction

            else:  # 默认策略
                # 随机逃跑方向
                direction = np.random.randn(2)
        else:
            # 没有威胁时随机移动
            direction = np.random.randn(2)

        self.move(direction)


class PursuitEvasionSimulation:
    """多机器人追逃模拟类"""

    def __init__(self, num_pursuers=3, num_evaders=5, boundary=(0, 100, 0, 100),
                 pursuer_speed=1.5, evader_speed=1.0, capture_distance=5.0,
                 enable_sharing=True):
        """初始化模拟环境"""
        self.boundary = boundary
        self.capture_distance = capture_distance
        self.active_simulation = False
        self.enable_sharing = enable_sharing

        # 创建L型障碍物
        self.obstacles = self.create_l_shaped_obstacle(boundary)

        # 创建地图探索器（传入障碍物信息）
        self.map_explorer = MapExplorer(boundary, grid_size=15, obstacles=self.obstacles)

        # 创建共享信息对象
        self.shared_info = SharedInformation()

        # 初始化追踪者
        self.pursuers = []
        for i in range(num_pursuers):
            position = self.find_valid_spawn_position(boundary)
            strategy = random.choice(['nearest', 'random'])
            sensor_range = random.uniform(20.0, 40.0)
            speed = random.uniform(0.8 * pursuer_speed, 1.2 * pursuer_speed)
            self.pursuers.append(
                Pursuer(position[0], position[1], speed, 3.0, strategy, sensor_range,
                        boundary, id=f"P{i}", shared_info=self.shared_info,
                        obstacles=self.obstacles, map_explorer=self.map_explorer)
            )

        # 初始化逃避者
        self.evaders = []
        for i in range(num_evaders):
            position = self.find_valid_spawn_position(boundary)
            strategy = random.choice(['avoid_nearest', 'avoid_all'])
            sensor_range = random.uniform(15.0, 30.0)
            speed = random.uniform(0.8 * evader_speed, 1.2 * evader_speed)
            self.evaders.append(
                Evader(position[0], position[1], speed, 2.0, strategy, sensor_range,
                       boundary, id=f"E{i}", shared_info=self.shared_info, obstacles=self.obstacles)
            )

        # 统计信息
        self.steps = 0
        self.captures = 0
        self.initial_evader_count = len(self.evaders)  # 记录初始逃避者数量
        self.history = []  # 存储历史状态

    def create_l_shaped_obstacle(self, boundary):
        """在地图中央创建L型障碍物"""
        min_x, max_x, min_y, max_y = boundary
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        obstacles = []

        # L型障碍物由两个矩形组成
        # 水平部分
        horizontal_width = 30
        horizontal_height = 8
        horizontal_x = center_x - horizontal_width / 2
        horizontal_y = center_y - horizontal_height / 2

        obstacles.append(Obstacle(horizontal_x, horizontal_y, horizontal_width, horizontal_height, 'rectangle'))

        # 垂直部分
        vertical_width = 8
        vertical_height = 25
        vertical_x = center_x - horizontal_width / 2
        vertical_y = center_y - horizontal_height / 2

        obstacles.append(Obstacle(vertical_x, vertical_y, vertical_width, vertical_height, 'rectangle'))

        return obstacles

    def find_valid_spawn_position(self, boundary, max_attempts=50):
        """寻找一个有效的生成位置（不与障碍物重叠）"""
        min_x, max_x, min_y, max_y = boundary

        for attempt in range(max_attempts):
            x = random.uniform(min_x + 5, max_x - 5)
            y = random.uniform(min_y + 5, max_y - 5)
            position = np.array([x, y])

            # 检查是否与障碍物重叠
            valid = True
            for obstacle in self.obstacles:
                if obstacle.distance_to_point(position) < 8.0:  # 最小间距
                    valid = False
                    break

            if valid:
                return position

        # 如果找不到有效位置，返回一个边角位置
        return np.array([min_x + 10, min_y + 10])

    def step(self):
        """模拟单步前进"""
        if not self.active_simulation:
            return

        # 清理过期的信息
        if self.enable_sharing:
            self.shared_info.clean_old_information(self.steps)

        # 更新所有追踪者
        for pursuer in self.pursuers:
            other_pursuers = [p for p in self.pursuers if p != pursuer]
            pursuer.update(self.evaders, other_pursuers, self.steps)

        # 更新所有逃避者
        for evader in self.evaders:
            other_evaders = [e for e in self.evaders if e != evader]
            evader.update(self.pursuers, other_evaders, self.steps)

        # 检查捕获并移除被捕获的逃避者
        evaders_to_remove = []
        for evader in self.evaders:
            for pursuer in self.pursuers:
                if evader.is_captured(pursuer, self.capture_distance):
                    evaders_to_remove.append(evader)
                    self.captures += 1

                    # 从共享信息中移除这个逃避者
                    self.shared_info.remove_evader_info(evader.id)

                    # 重置捕获到这个逃避者的追踪者的目标
                    for p in self.pursuers:
                        if p.target_id == evader.id:
                            p.target = None
                            p.target_id = None

                    break

        # 实际移除被捕获的逃避者
        for evader in evaders_to_remove:
            self.evaders.remove(evader)

        self.steps += 1

        # 保存历史状态（可选，用于回放）
        self.save_state()

        # 检查是否所有逃避者都被捕获
        return len(self.evaders) == 0

    def save_state(self):
        """保存当前状态用于回放"""
        state = {
            'step': self.steps,
            'pursuers': [{'position': p.position.copy(), 'color': p.color, 'id': p.id}
                         for p in self.pursuers],
            'evaders': [{'position': e.position.copy(), 'color': e.color, 'id': e.id}
                        for e in self.evaders]
        }
        self.history.append(state)

    def reset(self):
        """重置模拟"""
        self.steps = 0
        self.captures = 0
        self.history = []

        # 重置地图探索器
        if self.map_explorer:
            self.map_explorer.reset()

        # 重置共享信息
        self.shared_info = SharedInformation()

        # 重置所有追踪者的位置和状态
        for pursuer in self.pursuers:
            new_position = self.find_valid_spawn_position(self.boundary)
            pursuer.position = new_position
            pursuer.history = [pursuer.position.copy()]
            pursuer.shared_info = self.shared_info
            pursuer.target = None
            pursuer.target_id = None
            pursuer.exploration_mode = False
            pursuer.exploration_target = None
            pursuer.obstacles = self.obstacles

        # 重新创建逃避者
        self.evaders = []
        for i in range(self.initial_evader_count):
            position = self.find_valid_spawn_position(self.boundary)
            strategy = random.choice(['avoid_nearest', 'avoid_all'])
            sensor_range = random.uniform(15.0, 30.0)
            speed = random.uniform(0.8, 1.2)  # 使用默认速度范围
            self.evaders.append(
                Evader(position[0], position[1], speed, 2.0, strategy, sensor_range,
                       self.boundary, id=f"E{i}", shared_info=self.shared_info, obstacles=self.obstacles)
            )

    def draw(self, ax):
        """在matplotlib轴上绘制当前状态"""
        # 清除轴
        ax.clear()

        # 设置边界
        ax.set_xlim(self.boundary[0], self.boundary[1])
        ax.set_ylim(self.boundary[2], self.boundary[3])

        # 绘制边界
        rect = patches.Rectangle(
            (self.boundary[0], self.boundary[2]),
            self.boundary[1] - self.boundary[0],
            self.boundary[3] - self.boundary[2],
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)

        # 绘制L型障碍物
        for obstacle in self.obstacles:
            obstacle.draw(ax)

        # 绘制探索网格（可选，用于调试）
        if self.map_explorer and hasattr(self, 'show_exploration_grid') and self.show_exploration_grid:
            self.draw_exploration_grid(ax)

        # 检查是否所有追踪者都没有目标（地图探索模式）
        all_have_no_target = all(p.target is None for p in self.pursuers)

        # 绘制所有追踪者
        for pursuer in self.pursuers:
            pursuer.draw(ax)

            # 绘制传感器范围
            sensor_circle = plt.Circle(
                pursuer.position,
                pursuer.sensor_range,
                color='red',
                alpha=0.1
            )
            ax.add_artist(sensor_circle)

            # 绘制与目标的连线（如果有目标）
            if pursuer.target and pursuer.target in self.evaders:
                # 检查视线是否被障碍物阻挡
                if pursuer.can_see(pursuer.target):
                    # 直接感知的目标用实线
                    if pursuer.distance_to(pursuer.target) <= pursuer.sensor_range:
                        ax.plot([pursuer.position[0], pursuer.target.position[0]],
                                [pursuer.position[1], pursuer.target.position[1]],
                                'r-', alpha=0.6, linewidth=1)
                    else:
                        # 通过共享信息获知的目标用虚线
                        ax.plot([pursuer.position[0], pursuer.target.position[0]],
                                [pursuer.position[1], pursuer.target.position[1]],
                                'r--', alpha=0.4, linewidth=1)
                else:
                    # 视线被阻挡用点线
                    ax.plot([pursuer.position[0], pursuer.target.position[0]],
                            [pursuer.position[1], pursuer.target.position[1]],
                            'r:', alpha=0.3, linewidth=1)

            # 显示探索模式和目标
            if pursuer.exploration_mode and pursuer.exploration_target is not None:
                # 绘制探索目标
                ax.plot(pursuer.exploration_target[0], pursuer.exploration_target[1],
                        'yo', markersize=8, alpha=0.7)

                # 绘制到探索目标的线
                ax.plot([pursuer.position[0], pursuer.exploration_target[0]],
                        [pursuer.position[1], pursuer.exploration_target[1]],
                        'y--', alpha=0.5, linewidth=1)

                # 检查当前路径是否被阻挡
                path_blocked = False
                if self.map_explorer:
                    path_blocked = self.map_explorer.is_path_blocked(pursuer.position, pursuer.exploration_target)

                # 在追踪者上方显示其ID和模式
                mode_text = f'{pursuer.id}-探索'
                if path_blocked:
                    mode_text += ' (路径阻挡)'

                ax.text(pursuer.position[0], pursuer.position[1] + 8,
                        mode_text, color='black', ha='center', va='center',
                        fontsize=8, bbox=dict(facecolor='yellow' if not path_blocked else 'orange', alpha=0.7, pad=1))
            elif pursuer.target is None:
                # 没有目标时显示搜索状态
                ax.text(pursuer.position[0], pursuer.position[1] + 8,
                        f'{pursuer.id}-搜索', color='black', ha='center', va='center',
                        fontsize=8, bbox=dict(facecolor='orange', alpha=0.7, pad=1))

        # 绘制所有逃避者
        for evader in self.evaders:
            evader.draw(ax)

            # 绘制传感器范围
            sensor_circle = plt.Circle(
                evader.position,
                evader.sensor_range,
                color='blue',
                alpha=0.1
            )
            ax.add_artist(sensor_circle)

            # 绘制逃避者感知到的威胁
            if self.enable_sharing:
                threats = evader.get_threats(self.pursuers, self.steps)
                for threat in threats:
                    # 检查视线是否被阻挡
                    threat_position = threat['position']
                    if evader.can_see(threat['pursuer']):
                        if threat.get('directly_detected', False):
                            # 直接感知的威胁用实线
                            ax.plot([evader.position[0], threat_position[0]],
                                    [evader.position[1], threat_position[1]],
                                    'b-', alpha=0.3, linewidth=0.5)
                        else:
                            # 通过共享信息获知的威胁用虚线
                            ax.plot([evader.position[0], threat_position[0]],
                                    [evader.position[1], threat_position[1]],
                                    'b--', alpha=0.2, linewidth=0.5)
                    else:
                        # 视线被阻挡用点线
                        ax.plot([evader.position[0], threat_position[0]],
                                [evader.position[1], threat_position[1]],
                                'b:', alpha=0.1, linewidth=0.5)

        # 添加状态信息
        remaining_count = len(self.evaders)
        captured_count = self.initial_evader_count - remaining_count

        title = 'Pursuer-Evader Model with Smart Path Planning - Step: {}'.format(self.steps)
        if self.enable_sharing:
            title += ' (Information Sharing Enabled)'
        else:
            title += ' (Information Sharing Disabled)'

        if all_have_no_target:
            title += ' - Map Exploration Mode'

        ax.set_title(title)

        # 计算探索进度
        exploration_progress = 0
        if self.map_explorer:
            exploration_progress = self.map_explorer.get_exploration_progress()

        # 统计被阻挡的路径数量
        blocked_paths_count = 0
        if self.map_explorer:
            for pursuer_id, blocked_targets in self.map_explorer.blocked_targets.items():
                blocked_paths_count += len(blocked_targets)

        status_text = f'Remaining Evaders: {remaining_count}/{self.initial_evader_count}\n' \
                      f'Captured: {captured_count}\n' \
                      f'Map Explored: {exploration_progress:.1%}\n' \
                      f'Blocked Paths: {blocked_paths_count}'
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 添加图例
        pursuer_patch = patches.Patch(color='red', label='Pursuer')
        evader_patch = patches.Patch(color='blue', label='Evader')
        obstacle_patch = patches.Patch(color='gray', label='L-shaped Obstacle')
        exploration_patch = patches.Patch(color='yellow', label='Exploration Target')

        # 添加共享信息指示
        if self.enable_sharing:
            direct_line = plt.Line2D([0], [0], color='red', lw=1, alpha=0.6, label='Direct Detection')
            shared_line = plt.Line2D([0], [0], color='red', ls='--', lw=1, alpha=0.4, label='Shared Info')
            blocked_line = plt.Line2D([0], [0], color='red', ls=':', lw=1, alpha=0.3, label='Blocked Sight')
            ax.legend(handles=[pursuer_patch, evader_patch, obstacle_patch, exploration_patch,
                               direct_line, shared_line, blocked_line],
                      loc='upper right')
        else:
            ax.legend(handles=[pursuer_patch, evader_patch, obstacle_patch, exploration_patch],
                      loc='upper right')

    def draw_exploration_grid(self, ax):
        """绘制探索网格（调试用）"""
        if not self.map_explorer:
            return

        min_x, max_x, min_y, max_y = self.boundary
        grid_size = self.map_explorer.grid_size

        for grid_x in range(self.map_explorer.grid_width):
            for grid_y in range(self.map_explorer.grid_height):
                world_pos = self.map_explorer.get_world_position(grid_x, grid_y)

                if self.map_explorer.explored_grid[grid_x, grid_y]:
                    # 已探索区域用绿色半透明方块标记
                    rect = patches.Rectangle(
                        (world_pos[0] - grid_size / 2, world_pos[1] - grid_size / 2),
                        grid_size, grid_size,
                        facecolor='green', alpha=0.1, edgecolor='none'
                    )
                    ax.add_patch(rect)

        # 绘制被阻挡的目标点
        for pursuer_id, blocked_targets in self.map_explorer.blocked_targets.items():
            for target_tuple in blocked_targets:
                target_pos = np.array(target_tuple)
                rect = patches.Rectangle(
                    (target_pos[0] - grid_size / 2, target_pos[1] - grid_size / 2),
                    grid_size, grid_size,
                    facecolor='red', alpha=0.2, edgecolor='red', linewidth=1
                )
                ax.add_patch(rect)


def create_interactive_simulation():
    """创建交互式追逃模拟"""
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9)

    # 初始参数
    initial_num_pursuers = 4
    initial_num_evaders = 2
    initial_boundary = (0, 100, 0, 100)
    initial_pursuer_speed = 1.5
    initial_evader_speed = 1.0
    initial_capture_distance = 5.0
    initial_sharing_enabled = True

    # 创建模拟
    simulation = PursuitEvasionSimulation(
        num_pursuers=initial_num_pursuers,
        num_evaders=initial_num_evaders,
        boundary=initial_boundary,
        pursuer_speed=initial_pursuer_speed,
        evader_speed=initial_evader_speed,
        capture_distance=initial_capture_distance,
        enable_sharing=initial_sharing_enabled
    )

    # 添加调试开关
    simulation.show_exploration_grid = False

    # 绘制初始状态
    simulation.draw(ax)

    # 创建滑块和按钮
    # 创建滑块轴
    ax_pursuers = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_evaders = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_pursuer_speed = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_evader_speed = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_capture_distance = plt.axes([0.25, 0.05, 0.65, 0.03])

    # 创建滑块
    s_pursuers = Slider(ax_pursuers, 'Pursuers', 1, 10, valinit=initial_num_pursuers, valstep=1)
    s_evaders = Slider(ax_evaders, 'Evaders', 1, 20, valinit=initial_num_evaders, valstep=1)
    s_pursuer_speed = Slider(ax_pursuer_speed, 'Pursuer Speed', 0.5, 3.0, valinit=initial_pursuer_speed)
    s_evader_speed = Slider(ax_evader_speed, 'Evader Speed', 0.5, 3.0, valinit=initial_evader_speed)
    s_capture_distance = Slider(ax_capture_distance, 'Capture Distance', 1.0, 10.0, valinit=initial_capture_distance)

    # 创建按钮
    ax_reset = plt.axes([0.7, 0.30, 0.1, 0.04])
    ax_start_stop = plt.axes([0.55, 0.30, 0.1, 0.04])
    ax_step = plt.axes([0.4, 0.30, 0.1, 0.04])
    ax_toggle_sharing = plt.axes([0.25, 0.30, 0.1, 0.04])
    ax_toggle_grid = plt.axes([0.1, 0.30, 0.1, 0.04])

    button_reset = Button(ax_reset, 'Reset')
    button_start_stop = Button(ax_start_stop, 'Start')
    button_step = Button(ax_step, 'Step')
    button_toggle_sharing = Button(ax_toggle_sharing, 'Sharing: ON' if initial_sharing_enabled else 'Sharing: OFF')
    button_toggle_grid = Button(ax_toggle_grid, 'Grid: OFF')

    # 更新函数
    def update(val):
        # 获取滑块值
        num_pursuers = int(s_pursuers.val)
        num_evaders = int(s_evaders.val)
        pursuer_speed = s_pursuer_speed.val
        evader_speed = s_evader_speed.val
        capture_distance = s_capture_distance.val

        # 创建新的模拟
        nonlocal simulation
        old_sharing = simulation.enable_sharing
        old_grid = simulation.show_exploration_grid

        simulation = PursuitEvasionSimulation(
            num_pursuers=num_pursuers,
            num_evaders=num_evaders,
            boundary=initial_boundary,
            pursuer_speed=pursuer_speed,
            evader_speed=evader_speed,
            capture_distance=capture_distance,
            enable_sharing=old_sharing
        )

        simulation.show_exploration_grid = old_grid

        # 更新图形
        simulation.draw(ax)
        fig.canvas.draw_idle()

    # 单步事件
    def on_step(event):
        simulation.step()
        simulation.draw(ax)
        fig.canvas.draw_idle()

    # 重置事件
    def on_reset(event):
        simulation.reset()
        simulation.draw(ax)
        button_start_stop.label.set_text('Start')
        simulation.active_simulation = False
        fig.canvas.draw_idle()

    # 开始/停止事件
    def on_start_stop(event):
        simulation.active_simulation = not simulation.active_simulation
        if simulation.active_simulation:
            button_start_stop.label.set_text('Pause')
            # 开始动画
            ani.event_source.start()
        else:
            button_start_stop.label.set_text('Start')
            # 停止动画
            ani.event_source.stop()
        fig.canvas.draw_idle()

    # 切换信息共享事件
    def on_toggle_sharing(event):
        simulation.enable_sharing = not simulation.enable_sharing
        button_toggle_sharing.label.set_text('Sharing: ON' if simulation.enable_sharing else 'Sharing: OFF')
        fig.canvas.draw_idle()

    # 切换网格显示事件
    def on_toggle_grid(event):
        simulation.show_exploration_grid = not simulation.show_exploration_grid
        button_toggle_grid.label.set_text('Grid: ON' if simulation.show_exploration_grid else 'Grid: OFF')
        simulation.draw(ax)
        fig.canvas.draw_idle()

    # 连接事件
    s_pursuers.on_changed(update)
    s_evaders.on_changed(update)
    s_pursuer_speed.on_changed(update)
    s_evader_speed.on_changed(update)
    s_capture_distance.on_changed(update)

    button_reset.on_clicked(on_reset)
    button_start_stop.on_clicked(on_start_stop)
    button_step.on_clicked(on_step)
    button_toggle_sharing.on_clicked(on_toggle_sharing)
    button_toggle_grid.on_clicked(on_toggle_grid)

    # 创建动画
    def animate(i):
        if simulation.active_simulation:
            if len(simulation.evaders) == 0:
                # 所有逃避者都被捕获，停止模拟
                simulation.active_simulation = False
                button_start_stop.label.set_text('Start')
            else:
                simulation.step()
                simulation.draw(ax)
                fig.canvas.draw_idle()
        return ax,

    ani = animation.FuncAnimation(fig, animate, frames=None,
                                  interval=100, blit=False)

    plt.show()
    return simulation


# 运行模拟
if __name__ == "__main__":
    simulation = create_interactive_simulation()