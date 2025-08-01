import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import time
import csv
import os
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == '__main__':
    # 定义训练的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 准备数据集 - 添加数据增强
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 正确的数据集初始化
    train_data = torchvision.datasets.CIFAR10(root="./data", train=True, 
                                             transform=train_transform, download=True)
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False, 
                                             transform=test_transform, download=True)

    # 数据集长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print(f"训练数据集长度: {train_data_size}")
    print(f"测试数据集长度: {test_data_size}")

    # 利用DataLoader加载数据集 - 禁用多进程 (num_workers=0)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)

    # 优化后的网络模型
    class CifarNet(nn.Module):
        def __init__(self):
            super(CifarNet, self).__init__()
            
            self.features = nn.Sequential(
                # 卷积块1: 增加通道数，使用3x3卷积核
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.2),
                
                # 卷积块2
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.3),
                
                # 卷积块3
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.4),
                
                # 卷积块4
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.5),
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 10)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = CifarNet().to(device)

    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 训练参数
    total_train_step = 0
    epochs = 128
    start_time = time.time()
    best_acc = 0.0

    # 创建日志目录
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)

    # 创建CSV日志文件
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_log_{current_time}.csv")

    # 写入CSV表头
    with open(log_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Accuracy', 'Learning Rate', 'Epoch Time (s)', 'Total Time (s)'])

    print(f"训练日志将保存至: {log_filename}")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n--------- 第 {epoch+1}/{epochs} 轮训练开始 ---------")
        
        # 训练阶段
        model.train()
        train_losses = []
        
        for imgs, targets in train_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            train_losses.append(loss.item())
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # 计算平均训练损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # 测试阶段
        model.eval()
        test_losses = []
        total_correct = 0
        
        with torch.no_grad():
            for imgs, targets in test_dataloader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                
                loss = loss_fn(outputs, targets)
                test_losses.append(loss.item())
                
                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
        
        # 计算测试指标
        avg_test_loss = sum(test_losses) / len(test_losses)
        accuracy = total_correct / test_data_size
        
        # 更新学习率
        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算时间
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        # 打印训练结果
        print(f"训练损失: {avg_train_loss:.4f} | 测试损失: {avg_test_loss:.4f}")
        print(f"测试准确率: {accuracy:.4f} | 学习率: {current_lr:.6f}")
        print(f"本轮耗时: {epoch_time:.2f}秒 | 累计耗时: {total_time:.2f}秒")
        
        # 保存最佳模型
        if accuracy > best_acc:
            # 如果有旧的最佳模型存在，则删除它
            if best_acc > 0:  # 确保不是第一次保存
                old_model_path = os.path.join(log_dir, f"best_model_acc_{best_acc:.4f}.pth")
                if os.path.exists(old_model_path):
                    try:
                        os.remove(old_model_path)
                        print(f"已删除旧的最佳模型: {old_model_path}")
                    except Exception as e:
                        print(f"删除旧模型失败: {e}")
            
            # 更新最佳准确率并保存新模型
            best_acc = accuracy
            model_path = os.path.join(log_dir, f"best_model_acc_{best_acc:.4f}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"最佳模型已保存: {model_path}")
        
        
        # 写入CSV日志
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch+1, 
                f"{avg_train_loss:.6f}", 
                f"{avg_test_loss:.6f}", 
                f"{accuracy:.6f}",
                f"{current_lr:.6f}",
                f"{epoch_time:.2f}", 
                f"{total_time:.2f}"
            ])

    print("\n训练完成!")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    print(f"最佳准确率: {best_acc:.4f}")
    print(f"最终模型和训练日志保存在目录: {log_dir}")