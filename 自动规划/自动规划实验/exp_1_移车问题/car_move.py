import tkinter as tk
from collections import deque
import time
from PIL import Image, ImageTk


def is_valid(state, max_capacity):
    return all(len(garage) <= max_capacity for garage in state)


def move_car(state, from_idx, to_idx):
    new_state = [list(garage) for garage in state]
    car = new_state[from_idx].pop()
    new_state[to_idx].append(car)
    return tuple(tuple(garage) for garage in new_state)


def bfs(initial_state, target_state, max_capacity):
    queue = deque([(initial_state, [])])
    visited = set()
    visited.add(initial_state)

    while queue:
        current_state, path = queue.popleft()
        if current_state == target_state:
            return path

        for i in range(3):
            if len(current_state[i]) > 0:
                car = current_state[i][-1]
                for j in range(3):
                    if i != j and len(current_state[j]) < max_capacity:
                        new_state = move_car(current_state, i, j)
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append((new_state, path + [(car, i, j)]))
    return None


class CarParkingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("车库移车演示")
        self.root.geometry("800x600")

        self.canvas = tk.Canvas(root, width=800, height=400, bg="white")
        self.canvas.pack()

        self.input_frame = tk.Frame(root)
        self.input_frame.pack()

        self.garage_entries = []
        self.target_entries = []

        for i in range(3):
            tk.Label(self.input_frame, text=f"初始车库{i + 1}:").grid(row=i, column=0)
            entry = tk.Entry(self.input_frame, width=20)
            entry.grid(row=i, column=1)
            self.garage_entries.append(entry)

        for i in range(3):
            tk.Label(self.input_frame, text=f"目标车库{i + 1}:").grid(row=i, column=2)
            entry = tk.Entry(self.input_frame, width=20)
            entry.grid(row=i, column=3)
            self.target_entries.append(entry)

        tk.Label(self.input_frame, text="车库最大容量:").grid(row=3, column=0)
        self.capacity_entry = tk.Entry(self.input_frame, width=5)
        self.capacity_entry.grid(row=3, column=1)

        self.solve_button = tk.Button(root, text="求解并演示", command=self.solve)
        self.solve_button.pack()

        self.garage_img = ImageTk.PhotoImage(Image.open("garage.png").resize((200, 300)))
        self.car_img = Image.open("car.png").resize((80, 40))

        self.state = ((), (), ())
        self.draw_state()

    def parse_input(self, entries):
        return tuple(tuple(entry.get().split()) for entry in entries)

    def draw_state(self):
        self.canvas.delete("all")
        self.car_images = {}
        garage_positions = [(50, 250), (300, 250), (550, 250)]

        for i, garage in enumerate(self.state):
            x, y = garage_positions[i]
            self.canvas.create_image(x + 100, y, image=self.garage_img)
            for j, car in enumerate(garage):
                car_img_with_text = self.car_img.copy()
                car_tk = ImageTk.PhotoImage(car_img_with_text)
                self.car_images[car] = car_tk
                self.canvas.create_image(x + 100, y + 50 - j * 50, image=car_tk)
                self.canvas.create_text(x + 100, y + 50 - j * 50, text=car, font=("Arial", 20), fill="black")

    def animate_solution(self, solution):
        for car, from_idx, to_idx in solution:
            self.state = move_car(self.state, from_idx, to_idx)
            self.draw_state()
            self.root.update()
            time.sleep(1)

    def solve(self):
        try:
            self.state = self.parse_input(self.garage_entries)
            target_state = self.parse_input(self.target_entries)
            max_capacity = int(self.capacity_entry.get())
            solution = bfs(self.state, target_state, max_capacity)
            if solution:
                self.animate_solution(solution)
            else:
                print("没有找到解决方案")
        except Exception as e:
            print(f"输入错误: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CarParkingGUI(root)
    root.mainloop()
