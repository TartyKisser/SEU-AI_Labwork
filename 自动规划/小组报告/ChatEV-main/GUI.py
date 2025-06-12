import torch
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from datetime import datetime
import os
import sys
import traceback
# packages for data processing
import utils
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline


class EVChargingPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EV Charging Occupancy Predictor")
        self.root.geometry("900x700")

        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Arial", 11))
        self.style.configure("TButton", font=("Arial", 11))
        self.style.configure("TEntry", font=("Arial", 11))

        # 主框架
        self.main_frame = ttk.Frame(root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(self.main_frame, text="Electric Vehicle Charging Occupancy Predictor",
                  font=("Arial", 16, "bold")).pack(pady=10)

        # 输入区域框架
        input_frame = ttk.LabelFrame(self.main_frame, text="Input Parameters", padding=10)
        input_frame.pack(fill=tk.X, pady=10)

        # Zone选择
        zone_frame = ttk.Frame(input_frame)
        zone_frame.pack(fill=tk.X, pady=5)
        ttk.Label(zone_frame, text="Zone:").pack(side=tk.LEFT, padx=5)

        # 创建区域选择的下拉菜单
        self.zone_var = tk.StringVar()
        self.zone_combo = ttk.Combobox(zone_frame, textvariable=self.zone_var, width=10)
        self.zone_combo.pack(side=tk.LEFT, padx=5)

        # 时间输入
        time_frame = ttk.Frame(input_frame)
        time_frame.pack(fill=tk.X, pady=5)
        ttk.Label(time_frame, text="Time (YYYY-MM-DD HH:MM:SS):").pack(side=tk.LEFT, padx=5)
        self.time_var = tk.StringVar(value="2022-10-13 15:00:00")  # 默认时间
        time_entry = ttk.Entry(time_frame, textvariable=self.time_var, width=25)
        time_entry.pack(side=tk.LEFT, padx=5)

        # 按钮区域
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # 预测按钮
        self.predict_button = ttk.Button(button_frame, text="Generate Prediction", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        # 清除按钮
        self.clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_output)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # 状态标签
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(button_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=5)

        # 输出区域
        output_frame = ttk.LabelFrame(self.main_frame, text="Model Prompt", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 提示文本显示区域
        self.prompt_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10, font=("Courier", 10))
        self.prompt_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # 预测结果区域
        result_frame = ttk.LabelFrame(self.main_frame, text="Prediction Results", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 预测值 - 使用文本框以显示更多内容
        pred_frame = ttk.Frame(result_frame)
        pred_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Label(pred_frame, text="Predicted Occupancy:", font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=5, pady=2)

        # 使用文本框而不是标签来显示预测结果
        self.pred_text = scrolledtext.ScrolledText(pred_frame, wrap=tk.WORD, height=5, font=("Courier", 10))
        self.pred_text.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        self.pred_text.insert(tk.END, "-")

        # 真实值
        truth_frame = ttk.Frame(result_frame)
        truth_frame.pack(fill=tk.X, pady=5)
        ttk.Label(truth_frame, text="Ground Truth:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=5)
        self.truth_var = tk.StringVar(value="-")
        ttk.Label(truth_frame, textvariable=self.truth_var, font=("Arial", 11)).pack(side=tk.LEFT, padx=5)

        # 添加一个按钮来复制预测结果到剪贴板
        copy_button = ttk.Button(truth_frame, text="Copy Prediction",
                                 command=self.copy_prediction_to_clipboard)
        copy_button.pack(side=tk.RIGHT, padx=5)

        # 加载数据和模型
        self.load_data()

    def load_data(self):
        """加载数据和模型"""
        try:
            self.status_var.set("Loading data...")
            self.root.update()

            # 设置设备
            use_cuda = True
            self.device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

            # 加载数据
            self.occ, self.dur, self.vol, self.prc, self.adj, self.col, self.dis, self.weather, self.inf = utils.read_data()

            # 更新区域选择下拉菜单
            available_zones = list(range(self.occ.shape[1]))
            self.zone_combo['values'] = available_zones
            self.zone_var.set(str(42))  # 默认区域

            self.status_var.set("Data loaded. Ready to predict.")
        except Exception as e:
            error_msg = f"Failed to load data: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error loading data")

    def load_model(self):
        """加载模型和分词器"""
        try:
            # 确保离线模式设置
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"

            self.status_var.set("Loading model...")
            self.root.update()

            # 调用utils中的load_llm函数
            self.model, self.tokenizer, self.config = utils.load_llm()

            # 创建pipeline
            self.status_var.set("Creating text generation pipeline...")
            self.root.update()

            # 打印一些调试信息
            print(f"Model type: {type(self.model)}")
            print(f"Tokenizer type: {type(self.tokenizer)}")
            print(f"Device: {self.device}")

            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device
            )

            self.status_var.set("Model loaded successfully")
            return True
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error loading model")
            return False

    def time_to_index_optimized(self, time_str):
        """将时间字符串转换为CSV数据中的行索引"""
        # 标准化时间格式
        try:
            if '/' in time_str:
                # 处理格式为 'YYYY/MM/DD HH:MM' 或 'YYYY/MM/DD HH:MM:SS'
                if len(time_str.split(':')) == 2:  # 没有秒
                    input_time = datetime.strptime(time_str, '%Y/%m/%d %H:%M')
                else:  # 有秒
                    input_time = datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
            else:
                # 处理格式为 'YYYY-MM-DD HH:MM' 或 'YYYY-MM-DD HH:MM:SS'
                if len(time_str.split(':')) == 2:  # 没有秒
                    input_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
                else:  # 有秒
                    input_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            error_msg = f"Time format error: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            return -1

        # 数据开始和结束时间
        start_time = datetime.strptime('2022-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime('2023-02-28 23:00:00', '%Y-%m-%d %H:%M:%S')

        # 检查输入时间是否在范围内
        if input_time < start_time or input_time > end_time:
            error_msg = f"Time {time_str} is out of data range ({start_time} to {end_time})"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            return -1

        # 计算时间差（小时数）
        time_diff = (input_time - start_time).total_seconds() / 3600

        # 根据时间差计算行索引（假设每小时一条数据，行索引从0开始）
        # 四舍五入到最接近的整数，以处理可能的秒数差异
        row_index = round(time_diff)

        return row_index

    def predict(self):
        """执行预测"""
        # 禁用按钮，防止重复点击
        self.predict_button.config(state=tk.DISABLED)

        # 在新线程中执行预测，避免GUI卡顿
        threading.Thread(target=self._predict_thread).start()

    def _predict_thread(self):
        """预测线程"""
        try:
            # 获取输入参数
            zone = int(self.zone_var.get())
            time_str = self.time_var.get()

            self.status_var.set("Converting time to index...")
            self.root.update()

            # 转换时间为索引
            timestamp = self.time_to_index_optimized(time_str)
            if timestamp == -1:
                self.status_var.set("Failed to convert time")
                self.predict_button.config(state=tk.NORMAL)
                return

            self.status_var.set(f"Generating prompt for zone {zone}, timestamp {timestamp}...")
            self.root.update()

            # 生成提示
            try:
                input_prompt = utils.prompting(zone, timestamp, self.inf, self.occ, self.prc, self.weather)
                target = utils.output_template(np.round(self.occ.iloc[timestamp + 6, zone], decimals=4))
            except Exception as e:
                error_msg = f"Error generating prompt: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", error_msg)
                self.status_var.set("Error generating prompt")
                self.predict_button.config(state=tk.NORMAL)
                return

            # 更新提示文本显示
            self.prompt_text.delete(1.0, tk.END)
            self.prompt_text.insert(tk.END, input_prompt)

            # 更新真实值
            self.truth_var.set(target)

            # 如果模型还没加载，加载模型
            if not hasattr(self, 'pipe'):
                self.status_var.set("Loading model...")
                self.root.update()
                if not self.load_model():
                    self.predict_button.config(state=tk.NORMAL)
                    return

            self.status_var.set("Generating prediction...")
            self.root.update()

            # 使用模型进行预测
            # 根据模型的不同，输入格式可能不同
            try:
                # 尝试使用messages格式（适用于聊天模型）
                messages = [
                    {"role": "user", "content": input_prompt}
                ]

                self.status_var.set("Running model inference...")
                self.root.update()

                outputs = self.pipe(
                    messages,
                    max_new_tokens=128,
                    do_sample=True
                )

                # 记录原始输出以便调试
                print("Raw model output:", outputs)

                # 获取结果 - 处理聊天格式输出
                full_prediction = ""
                clean_prediction = ""

                if isinstance(outputs, list) and len(outputs) > 0:
                    # 处理输出为列表的情况
                    if len(outputs) > 1 and "content" in outputs[1]:
                        # 如果是带有assistant回复的格式
                        assistant_content = outputs[1]["content"]
                        full_prediction = assistant_content

                        # 提取预测值部分
                        if "I predict the charging occupancy" in assistant_content:
                            try:
                                # 提取"I predict the charging occupancy..."开始的部分
                                start_idx = assistant_content.find("I predict the charging occupancy")
                                if start_idx != -1:
                                    clean_prediction = assistant_content[start_idx:]
                                else:
                                    clean_prediction = assistant_content
                            except:
                                clean_prediction = assistant_content
                        else:
                            clean_prediction = assistant_content

                    elif "generated_text" in outputs[0]:
                        generated_text = outputs[0]["generated_text"]
                        # 保存完整输出以便调试
                        full_prediction = str(generated_text)

                        # 如果generated_text是字符串，则可以使用split
                        if isinstance(generated_text, str):
                            # 尝试不同的分隔符
                            if "### RESPONSE:" in generated_text:
                                response_part = generated_text.split("### RESPONSE:")[-1].strip()
                                clean_prediction = response_part
                            elif "RESPONSE:" in generated_text:
                                response_part = generated_text.split("RESPONSE:")[-1].strip()
                                clean_prediction = response_part
                            # 检查是否包含"I predict the charging occupancy"
                            elif "I predict the charging occupancy" in generated_text:
                                start_idx = generated_text.find("I predict the charging occupancy")
                                if start_idx != -1:
                                    clean_prediction = generated_text[start_idx:]
                                else:
                                    clean_prediction = generated_text
                            else:
                                # 如果没有明确的分隔符，可能需要查找模型输出的最后部分
                                clean_prediction = generated_text
                        else:
                            # 如果不是字符串，尝试转换
                            clean_prediction = str(generated_text)
                    else:
                        # 输出没有预期的格式，直接使用
                        clean_prediction = str(outputs[0])
                        full_prediction = str(outputs)
                else:
                    # 如果输出不是列表或列表为空
                    clean_prediction = str(outputs)
                    full_prediction = str(outputs)

                # 清理预测结果，进一步处理
                # 如果包含特定文本，尝试精确提取预测数字
                if "I predict the charging occupancy" in clean_prediction:
                    lines = clean_prediction.split('\n')
                    result_lines = []

                    # 包含标题行
                    for i, line in enumerate(lines):
                        if "I predict the charging occupancy" in line:
                            result_lines.append(line)
                            # 收集后续的数值行
                            j = i + 1
                            while j < len(lines) and (lines[j].strip().replace(".", "").isdigit() or
                                                      any(c.isdigit() for c in lines[j])):
                                result_lines.append(lines[j])
                                j += 1
                            break

                    if result_lines:
                        clean_prediction = '\n'.join(result_lines)

            except Exception as e:
                error_msg = f"Error during prediction: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()

                # 尝试以不同的方式进行预测
                try:
                    self.status_var.set("Trying alternative prediction method...")
                    self.root.update()

                    # 尝试直接传入文本
                    outputs = self.pipe(
                        input_prompt,
                        max_new_tokens=128,
                        do_sample=True
                    )

                    print("Alternative method raw output:", outputs)

                    # 处理输出
                    if isinstance(outputs, list) and len(outputs) > 0:
                        if isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                            generated_text = outputs[0]["generated_text"]
                            full_prediction = str(generated_text)

                            if isinstance(generated_text, str):
                                if "### RESPONSE:" in generated_text:
                                    clean_prediction = generated_text.split("### RESPONSE:")[-1].strip()
                                elif "RESPONSE:" in generated_text:
                                    clean_prediction = generated_text.split("RESPONSE:")[-1].strip()
                                elif "I predict the charging occupancy" in generated_text:
                                    start_idx = generated_text.find("I predict the charging occupancy")
                                    if start_idx != -1:
                                        clean_prediction = generated_text[start_idx:]
                                    else:
                                        clean_prediction = generated_text
                                else:
                                    clean_prediction = generated_text
                            else:
                                clean_prediction = str(generated_text)
                        else:
                            clean_prediction = str(outputs[0])
                            full_prediction = str(outputs)
                    else:
                        clean_prediction = str(outputs)
                        full_prediction = str(outputs)

                    # 进一步清理预测结果
                    if "I predict the charging occupancy" in clean_prediction:
                        lines = clean_prediction.split('\n')
                        result_lines = []

                        for i, line in enumerate(lines):
                            if "I predict the charging occupancy" in line:
                                result_lines.append(line)
                                # 收集后续的数值行
                                j = i + 1
                                while j < len(lines) and (lines[j].strip().replace(".", "").isdigit() or
                                                          any(c.isdigit() for c in lines[j])):
                                    result_lines.append(lines[j])
                                    j += 1
                                break

                        if result_lines:
                            clean_prediction = '\n'.join(result_lines)

                except Exception as inner_e:
                    error_msg = f"First attempt failed: {str(e)}\nSecond attempt failed: {str(inner_e)}"
                    print(error_msg)
                    traceback.print_exc()
                    messagebox.showerror("Error", error_msg)
                    clean_prediction = "Error generating prediction"
                    full_prediction = error_msg

            # 显示预测结果
            self.status_var.set("Displaying prediction...")
            self.root.update()

            # 清除并设置预测文本
            self.pred_text.delete(1.0, tk.END)

            # 显示预测结果 - 使用清理后的预测内容
            if clean_prediction:
                self.pred_text.insert(tk.END, clean_prediction)
            else:
                self.pred_text.insert(tk.END, full_prediction)

            self.status_var.set("Prediction complete")

        except Exception as e:
            error_msg = f"Prediction process failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Prediction failed")
        finally:
            # 重新启用按钮
            self.predict_button.config(state=tk.NORMAL)

    def copy_prediction_to_clipboard(self):
        """复制预测结果到剪贴板"""
        prediction = self.pred_text.get(1.0, tk.END).strip()
        if prediction and prediction != "-":
            self.root.clipboard_clear()
            self.root.clipboard_append(prediction)
            self.status_var.set("Prediction copied to clipboard")
            self.root.update()

    def clear_output(self):
        """清除输出"""
        self.prompt_text.delete(1.0, tk.END)
        self.pred_text.delete(1.0, tk.END)
        self.pred_text.insert(tk.END, "-")
        self.truth_var.set("-")
        self.status_var.set("Ready")


# 主函数
def main():
    root = tk.Tk()
    app = EVChargingPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()