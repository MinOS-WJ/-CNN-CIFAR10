import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import os
import time
import numpy as np

# 确保与train.py相同的模型结构
class CifarNet(torch.nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        
        self.features = torch.nn.Sequential(
            # 卷积块1
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.2),
            
            # 卷积块2
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3),
            
            # 卷积块3
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.4),
            
            # 卷积块4
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.5),
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# CIFAR10类别标签
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# 全局变量
model = None
model_path = ""

# 加载模型函数
def load_model(path=None):
    global model, model_path
    
    if not path:
        path = filedialog.askopenfilename(
            title="选择CIFAR10模型文件",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )
        if not path:
            return
    
    try:
        # 创建新模型实例
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        new_model = CifarNet()
        
        # 加载权重
        state_dict = torch.load(path, map_location=device)
        new_model.load_state_dict(state_dict)
        new_model.to(device)
        new_model.eval()
        
        # 更新全局模型和路径
        model = new_model
        model_path = path
        
        # 更新UI
        model_status.config(text=f"模型已加载: {os.path.basename(path)}")
        btn_load_model.config(text="更换模型")
        messagebox.showinfo("成功", f"模型加载成功!\n路径: {path}")
        
        # 启用预测功能
        btn_load_image.config(state=tk.NORMAL)
        label_result.config(text="请上传图片进行预测")
        
        # 更新状态栏
        update_status(f"模型加载完成 | 设备: {device}")
        
    except Exception as e:
        messagebox.showerror("错误", f"加载模型失败: {e}")
        model_status.config(text="模型未加载")
        update_status(f"错误: {str(e)}")

# 图像预处理 (与train.py中的测试预处理一致)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 预测函数
def predict_image(image_path):
    global model
    if model is None:
        messagebox.showerror("错误", "请先加载模型")
        return None, None
    
    try:
        start_time = time.time()
        img = Image.open(image_path)
        
        # 修复通道数问题：转换为RGB格式（移除alpha通道）
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_tensor = transform(img).unsqueeze(0)  # 加入batch维度
        
        # 获取设备并将数据移动到相同设备
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
        
        # 计算概率
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted = torch.max(output, 1)
        
        # 获取预测结果和置信度
        class_index = predicted.item()
        confidence = probabilities[class_index].item()
        
        # 获取所有类别的概率
        all_probs = {CLASS_NAMES[i]: f"{probabilities[i].item()*100:.1f}%" 
                     for i in range(len(CLASS_NAMES))}
        
        inference_time = time.time() - start_time
        update_status(f"预测完成 | 耗时: {inference_time:.3f}秒 | 置信度: {confidence*100:.1f}%")
        
        return class_index, all_probs
    
    except Exception as e:
        messagebox.showerror("错误", f"预测失败: {e}")
        update_status(f"错误: {str(e)}")
        return None, None

# 创建概率显示窗口
def show_probabilities_window(probs):
    prob_window = tk.Toplevel(root)
    prob_window.title("分类概率分布")
    prob_window.geometry("300x400")
    
    frame = ttk.Frame(prob_window)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 标题
    ttk.Label(frame, text="类别概率分布", font=("Arial", 14, "bold")).pack(pady=10)
    
    # 创建滚动区域
    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # 显示每个类别的概率
    for i, (class_name, prob) in enumerate(probs.items()):
        row_frame = ttk.Frame(scrollable_frame)
        row_frame.pack(fill=tk.X, pady=2)
        
        # 类别标签
        ttk.Label(row_frame, text=class_name, width=15, anchor="w").pack(side=tk.LEFT)
        
        # 进度条
        prob_value = float(prob.strip('%')) / 100
        style = ttk.Style()
        style.configure(f"prob.Horizontal.TProgressbar", background='#4CAF50')
        pb = ttk.Progressbar(
            row_frame, 
            orient="horizontal", 
            length=150, 
            mode="determinate",
            style=f"prob.Horizontal.TProgressbar"
        )
        pb.pack(side=tk.LEFT, padx=5)
        pb['value'] = prob_value * 100
        
        # 概率值
        ttk.Label(row_frame, text=prob, width=8).pack(side=tk.LEFT)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

# 加载图像函数
def load_image():
    file_path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("图片文件", "*.png *.jpg *.jpeg"), ("所有文件", "*.*")]
    )
    
    if file_path:
        # 显示加载状态
        label_result.config(text="预测中...")
        preview_label.config(image='')
        btn_show_probs.config(state=tk.DISABLED)  # 禁用详情按钮
        root.update()  # 立即更新界面
        
        try:
            # 打开图像并转换为RGB格式（修复RGBA问题）
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 显示原始图片
            img_preview = img.copy()
            img_preview.thumbnail((300, 300))  # 调整预览大小
            img_preview = ImageTk.PhotoImage(img_preview)
            preview_label.config(image=img_preview)
            preview_label.image = img_preview  # 保持引用
            
            # 进行预测
            class_index, all_probs = predict_image(file_path)
            
            if class_index is not None:
                class_name = CLASS_NAMES[class_index]
                label_result.config(text=f"预测结果: {class_name}", 
                                  font=("Arial", 16, "bold"),
                                  foreground="#2E7D32")
                
                # 添加查看概率详情按钮
                btn_show_probs.config(state=tk.NORMAL, 
                                     command=lambda: show_probabilities_window(all_probs))
        except Exception as e:
            messagebox.showerror("错误", f"处理图像失败: {e}")
            update_status(f"错误: {str(e)}")

# 更新状态栏
def update_status(message):
    status_bar.config(text=message)
    status_bar.update()

# 创建主窗口
root = tk.Tk()
root.title("CIFAR10图像分类系统")
root.geometry("700x600")  # 增大窗口尺寸

# 设置样式
style = ttk.Style()
style.configure("TButton", font=("Arial", 11))
style.configure("TLabel", font=("Arial", 11))
style.configure("Status.TLabel", font=("Arial", 10), background="#E0E0E0")

# 创建主框架
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# 标题
label_title = ttk.Label(main_frame, text="CIFAR10图像分类系统", 
                      font=("Arial", 20, "bold"))
label_title.pack(pady=15)

# 模型区域框架
model_frame = ttk.LabelFrame(main_frame, text="模型管理")
model_frame.pack(fill=tk.X, pady=10)

# 模型加载按钮
btn_load_model = ttk.Button(model_frame, text="加载模型", command=load_model)
btn_load_model.pack(side=tk.LEFT, padx=10, pady=10)

# 模型状态标签
model_status = ttk.Label(model_frame, text="模型未加载", font=("Arial", 10))
model_status.pack(side=tk.LEFT, padx=10, pady=10)

# 图片区域框架
image_frame = ttk.LabelFrame(main_frame, text="图像预览")
image_frame.pack(fill=tk.BOTH, expand=True, pady=10)

# 图片预览标签
preview_label = ttk.Label(image_frame)
preview_label.pack(pady=20, padx=20)

# 操作按钮区域
button_frame = ttk.Frame(main_frame)
button_frame.pack(fill=tk.X, pady=10)

# 上传图片按钮
btn_load_image = ttk.Button(button_frame, text="上传图片", command=load_image, state=tk.DISABLED)
btn_load_image.pack(side=tk.LEFT, padx=5, pady=5)

# 查看概率按钮 (初始禁用)
btn_show_probs = ttk.Button(button_frame, text="查看详细概率", state=tk.DISABLED)
btn_show_probs.pack(side=tk.LEFT, padx=5, pady=5)

# 预测结果标签
label_result = ttk.Label(main_frame, text="请先加载模型", 
                       font=("Arial", 14), 
                       foreground="#616161")
label_result.pack(pady=15)

# 状态栏
status_bar = ttk.Label(root, text="就绪", relief=tk.SUNKEN, anchor=tk.W, 
                     style="Status.TLabel")
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# 尝试加载默认模型
default_model_dir = "training_logs"
if os.path.exists(default_model_dir):
    # 查找最新的模型文件
    model_files = [f for f in os.listdir(default_model_dir) if f.startswith("best_model_") and f.endswith(".pth")]
    if model_files:
        # 按准确率排序
        model_files.sort(key=lambda f: float(f.split('_')[-1][:-4]), reverse=True)
        best_model = os.path.join(default_model_dir, model_files[0])
        load_model(best_model)
        update_status(f"已自动加载最佳模型: {os.path.basename(best_model)}")

# 启动主循环
root.mainloop()