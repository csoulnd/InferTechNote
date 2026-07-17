# Why GPU so Fast？
希望通过这篇文章了解一下GPU的发展和和相关的硬件知识，帮助学习NPU的相关知识

## GPU的诞生

随着缩放定律带来的芯片性能提升走向瓶颈，工程师将视野转向专用硬件如TPU，然而，专用计算硬件只能聚焦于某一类或者某几类特定的计算任务，在处理其他任务时则可能力不从心。
而GPU则是向通用性演进的典型代表。虽然其最初设计目标是为图形渲染加速，但高度并行的SIMT（Single Instruction Multiple Threads，单指令多线程）架构意外契合了通用计算的演进需求，超高的并行度获得了远超CPU的计算性能。
<details close>
<summary>CPU VS GPU</summary>

1. CPU：少量大核心，强通用、强分支逻辑
CPU 核心数量很少（主流台式机 4~32 核），每个核心超大缓存、复杂控制单元、强大分支预测、乱序执行。
擅长：
复杂分支判断、if/else、循环跳转、递归、函数调用（复杂逻辑）
串行流程、条件多变、依赖前一步结果的计算
复杂数据结构：链表、树、哈希、复杂指针操作
分支多、逻辑不规则的任务（业务代码、操作系统、编译器）
缺点：同一时间能跑的独立计算流很少，大规模重复计算效率低。

2. GPU：海量小核心，弱逻辑、强并行
GPU 有成百上千个极简小计算核心（流处理器 CUDA Core / ALU），控制单元极简，缓存很小，分支处理能力很差。
设计目标：大量无依赖、重复、同一种运算同时执行。
擅长：
矩阵运算、图像像素处理、光线追踪、AI 训练推理
海量数据做相同数学计算（向量、浮点运算）
规则、少分支、数据互相不依赖的任务
短板：一旦出现大量分支判断（一部分线程走 A 逻辑、一部分走 B），会出现线程束分化，性能暴跌；复杂递归、复杂指针运算几乎不适合 GPU。
</details>

- GPU快的核心：[高并发计算](# "相比于CPU，单位面积内逻辑控制单元更少，流处理器更多")；[低内存延迟](# 'SIMT核心管理多个线程组（wrap）不会因为等待内存数据阻塞执行')；[特化内存与计算架构](# 'GPU常配备高带宽内存；GPU还会集成专用计算单元')
- 算力评估：FLOPS（Floating-Point Operations Per Second，每秒浮点运算次数）来表示，通常数量级为T(万亿)，也即是大家听到的TFLOPS，公式如下：

```
算力（FLOPS）= CUDA核心数 × 加速频率 × 每核心单个周期浮点计算系数
```
- GPU架构原型：[Fermi](https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf)架构是现代通用GPU架构的基石

### GPU 架构简图

以 Fermi 为代表的现代通用 GPU，整体可理解为「大量 SM + 共享 L2 + 高带宽显存」：

```mermaid
flowchart TB
    CPU["CPU（主机）"]

    subgraph GPU["GPU 芯片"]
        direction TB
        GPC["GPC × N<br/>图形处理集群"]
        SM["SM × 大量<br/>流式多处理器"]
        L2["L2 缓存"]
        MC["内存控制器"]

        GPC --> SM
        SM --> L2
        L2 --> MC
    end

    VRAM[("HBM / GDDR<br/>高带宽显存")]

    CPU <-->|PCIe| GPC
    MC <--> VRAM
```

SM 是 GPU 的基本计算单元，内部结构大致如下：

```mermaid
flowchart LR
    subgraph SM["SM（流式多处理器）"]
        direction TB
        WS["Warp 调度器<br/>（SIMT 执行）"]
        CORE["CUDA Core × 32<br/>浮点 / 整数 ALU"]
        REG["寄存器文件"]
        SHMEM["共享内存 / L1"]

        WS --> CORE
        CORE --- REG
        CORE --- SHMEM
    end
```

### CUDA编程简要示例

以CPU上的SAXPY为例，对比下两段代码
CPU:
'''
// SAXPY函数实现
void saxpy(int n, float a, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    float a = 2.0;
    int n; // 向量长度
    float *x; // 向量x
    float *y; // 向量y
    // 此处省略内存分配、元素赋值、长度指定
    // ...
    // 调用SAXPY函数
    saxpy(n, a, x, y);

    return 0;
}
'''
GPU:
'''
__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    float a = 2.0;
    int n; // 向量长度
    float *hx; // host向量x
    float *hy; // host向量y
    // 此处省略内存分配、元素赋值、长度指定
       
    // GPU内存分配
    int vector_size = n * sizeof(float); // 向量数据大小
    float *dx; // device向量x
    float *dy; // device向量y
    cudaMalloc(&dx, vector_size);
    cudaMalloc(&dy, vector_size);
    
    // 将host向量内容拷贝到device向量
    cudaMemcpy(dx, hx, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, vector_size, cudaMemcpyHostToDevice);
    
    // 执行saxpy
    int t = 256; // 每个thread block的线程数
    int blocks_num = (n + t - 1) / t; // thread block数量
    saxpy<<<blocks_num, t>>>(n, a, dx, dy);
    
    // 将device向量y内容（计算结果）拷贝到host向量y
    cudaMemcpy(hy, dy, vector_size, cudaMemcpyDeviceToHost);
    
    // ... (剩余逻辑)
    
    return 0;
}
'''
#### Host? Device!
GPU编程的思维是将GPU当作CPU的协同外设使用，通常GPU自身无法独立运行，需要CPU指定任务，分配数据，驱动运行。__gloable__代表内核函数，交给GPU来执行。
cudaMalloc、cudaMemcpy，是CPU操作GPU内存的操作,CPU分配GPU的内存用于GPU的数据搬运和计算，因此我们常说的Host,Device就由来于此。文中提到的统一内存（unified memory）代指CPU和GPU共享同一段地址空间的内存架构，理论上可以实现数据交换自动化，没有host to device但是事实上呢？

#### 线程组织
在CUDA编程中，线程以thread，thread block，grid的层级结构进行组织
● 线程（thread，绿色部分）：最基本的执行单元。线程包含独立寄存器状态和独立程序计数器。

● 线程块（thread block，黄色部分）：由多个线程组成的集合，支持一维、二维或三维结构。线程块内的线程可以通过共享内存进行通信，线程块之间无法通过共享内存通信，但可通过全局内存进行数据交互。

● Warp（蓝色线框）：硬件底层概念，GPU实际运行时将32个线程组成一个warp，同一warp内的线程同步执行相同的指令。

● 线程块与warp的关系：warp是底层概念，NVIDIA的warp固定包含32个线程，warp是线程硬件调度的最小粒度。线程块是软件概念，线程块有多少个线程组成由代码指定。在运行时，硬件会将线程块中的线程32个为一组打包成多个warp进行调度，因此，线程块里的线程数最好为32的整数倍，以避免为拼凑完整warp而自动分配无效线程造成资源浪费。

● 网格（grid，总体）：网格是所有线程块的集合，支持一维、二维或三维结构，覆盖整个计算任务的运行范围。
AMD叫法：NDRange；Work Group；Wavefront；Work Item
线程块计算：线程块包含的线程数可以指定，线程块的数量由计算规模确定，
'''
int blocks_num = (n + t - 1) / t; // thread block数量,向上取整写法
'''
## 参考文档

1. [知乎专栏](https://zhuanlan.zhihu.com/p/678001378)
2. [知乎专栏](https://zhuanlan.zhihu.com/p/31825598174)