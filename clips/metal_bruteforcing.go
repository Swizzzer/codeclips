// 爆破8字节消息的SHA-1
package main

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreGraphics

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Metal 设备和资源
id<MTLDevice> device;
id<MTLCommandQueue> commandQueue;
id<MTLComputePipelineState> computePipelineState;
id<MTLBuffer> resultBuffer;
id<MTLBuffer> targetBuffer;
id<MTLBuffer> foundBuffer;
id<MTLBuffer> baseIndexBuffer; // Renamed for clarity

// SHA-1 Metal shader 源码
const char* sha1MetalSource = R"(
#include <metal_stdlib>
using namespace metal;

// 循环左移
inline uint32_t rotateLeft(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// 字符映射
inline uchar indexToChar(uint64_t index) {
    // Alphabet: "0123456789abcdef"
    const uchar chars[16] = {
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    };
    return chars[index & 0xF]; // use mask for performance
}

// SHA-1 核心计算 - 使用线程本地内存
void sha1_hash_local(thread const uchar* input, thread uchar* output) {
    // SHA-1 初始化向量
    uint32_t H[5] = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    };

    // 准备消息块 (8-byte input)
    // Padding: input (8) + 0x80 (1) + zeros (47) + length (8) = 64 bytes
    uint32_t W[80];

    // Block 1 (only one block for 8-byte message)
    // Convert 8-byte input to two 32-bit words
    W[0] = ((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) | ((uint32_t)input[2] << 8) | input[3];
    W[1] = ((uint32_t)input[4] << 24) | ((uint32_t)input[5] << 16) | ((uint32_t)input[6] << 8) | input[7];
    W[2] = 0x80000000; // Padding
    W[3] = 0; W[4] = 0; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0; W[12] = 0;
    W[13] = 0;
    W[14] = 0; // Message length in bits (8 bytes * 8 bits = 64)
    W[15] = 64;

    // 消息调度
    for (int i = 16; i < 80; i++) {
        W[i] = rotateLeft(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1);
    }

    // 压缩函数
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4];
    uint32_t f, k;

    for (int i = 0; i < 80; i++) {
        if (i < 20) {
            f = (b & c) | ((~b) & d);
            k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }

        uint32_t temp = rotateLeft(a, 5) + f + e + k + W[i];
        e = d;
        d = c;
        c = rotateLeft(b, 30);
        b = a;
        a = temp;
    }

    // 更新哈希值
    H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e;

    // 输出大端序 (20 bytes)
    for (int i = 0; i < 5; i++) {
        output[i*4 + 0] = (H[i] >> 24) & 0xff;
        output[i*4 + 1] = (H[i] >> 16) & 0xff;
        output[i*4 + 2] = (H[i] >> 8)  & 0xff;
        output[i*4 + 3] = H[i]         & 0xff;
    }
}

// GPU 内核函数
kernel void sha1_search(
    device uchar* result_out [[buffer(0)]],      // 输出结果 (8 bytes)
    constant uchar* target [[buffer(1)]],        // 目标哈希 (20 bytes)
    device atomic_uint* found [[buffer(2)]],     // 找到标志
    constant uint64_t* baseIndex [[buffer(3)]],  // 基础索引
    uint gid [[thread_position_in_grid]]         // 线程ID
) {
    // 检查是否已找到
    if (atomic_load_explicit(found, memory_order_relaxed) != 0) {
        return;
    }

    uint64_t candidateIndex = baseIndex[0] + gid;

    // 生成 8 字节候选者
    thread uchar candidate[8];
    uint64_t idx = candidateIndex;
    // Generate from right to left for simplicity
    for (int i = 7; i >= 0; i--) {
        candidate[i] = indexToChar(idx);
        idx >>= 4; // Move to the next character
    }

    // 计算哈希
    thread uchar hash[20];
    sha1_hash_local(candidate, hash);

    // 比较结果
    bool match = true;
    for (int i = 0; i < 20; i++) {
        if (hash[i] != target[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        // 找到了！
        atomic_store_explicit(found, 1, memory_order_relaxed);

        // 保存结果到全局内存
        for (int i = 0; i < 8; i++) {
            result_out[i] = candidate[i];
        }
    }
}
)";

// 获取GPU信息
typedef struct {
    int coreCount;
    int maxThreadsPerThreadgroup;
    char name[256];
} GPUInfo;

// 使用system_profiler获取准确的GPU核心数
int getGPUCoresFromSystemProfiler() {
    FILE *fp;
    char buffer[128];
    int cores = 0;
    fp = popen("system_profiler SPDisplaysDataType | awk '/Total Number of Cores:/{print $5}'", "r");
    if (fp == NULL) { return 0; }
    if (fgets(buffer, sizeof(buffer), fp) != NULL) { cores = atoi(buffer); }
    pclose(fp);
    return cores;
}

GPUInfo getGPUInfo() {
    GPUInfo info = {0};
    if (device) {
        strncpy(info.name, [[device name] UTF8String], 255);
        info.coreCount = getGPUCoresFromSystemProfiler();
        if (info.coreCount == 0) {
            printf("Warning: Could not detect GPU cores via system_profiler, using a default value.\n");
            info.coreCount = 8; // Fallback
        }
    }
    return info;
}

// 初始化 Metal
int initMetal(GPUInfo* gpuInfo) {
    @autoreleasepool {
        NSError *error = nil;
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("Metal is not supported on this device\n");
            return -1;
        }

        *gpuInfo = getGPUInfo();
        printf("\n=== GPU Information ===\n");
        printf("GPU: %s\n", gpuInfo->name);
        printf("GPU Cores: %d\n", gpuInfo->coreCount);

        commandQueue = [device newCommandQueue];
        if (!commandQueue) { return -1; }

        NSString *source = [NSString stringWithUTF8String:sha1MetalSource];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            printf("Failed to compile shader: %s\n", [[error description] UTF8String]);
            return -1;
        }

        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"sha1_search"];
        if (!kernelFunction) { return -1; }

        computePipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!computePipelineState) {
            printf("Failed to create pipeline state: %s\n", [[error description] UTF8String]);
            return -1;
        }

        gpuInfo->maxThreadsPerThreadgroup = (int)computePipelineState.maxTotalThreadsPerThreadgroup;
        printf("Pipeline Max Threads Per Group: %d\n", gpuInfo->maxThreadsPerThreadgroup);

        // 创建缓冲区 (Update sizes for SHA-1)
        resultBuffer = [device newBufferWithLength:8 options:MTLResourceStorageModeShared];   // 8-byte result
        targetBuffer = [device newBufferWithLength:20 options:MTLResourceStorageModeShared];  // 20-byte SHA-1 hash
        foundBuffer = [device newBufferWithLength:sizeof(unsigned int) options:MTLResourceStorageModeShared];
        baseIndexBuffer = [device newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];

        if (!resultBuffer || !targetBuffer || !foundBuffer || !baseIndexBuffer) {
            printf("Failed to create buffers\n");
            return -1;
        }
        return 0;
    }
}

// 在GPU上搜索
int searchOnGPU(uint64_t startIndex, uint64_t count, const uint8_t* target, uint8_t* result, int maxThreadsPerThreadgroup) {
    @autoreleasepool {
        memcpy([targetBuffer contents], target, 20); // 20 bytes for SHA-1 target
        *(uint64_t*)[baseIndexBuffer contents] = startIndex;
        *(unsigned int*)[foundBuffer contents] = 0;

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:computePipelineState];
        [encoder setBuffer:resultBuffer offset:0 atIndex:0];
        [encoder setBuffer:targetBuffer offset:0 atIndex:1];
        [encoder setBuffer:foundBuffer offset:0 atIndex:2];
        [encoder setBuffer:baseIndexBuffer offset:0 atIndex:3];

        NSUInteger w = computePipelineState.threadExecutionWidth;
        NSUInteger threadsPerGroup = (maxThreadsPerThreadgroup / w) * w;
        if (threadsPerGroup == 0) {
             threadsPerGroup = w;
        }
        if (threadsPerGroup > computePipelineState.maxTotalThreadsPerThreadgroup) {
            threadsPerGroup = computePipelineState.maxTotalThreadsPerThreadgroup;
        }

        MTLSize threadsPerThreadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize grid = MTLSizeMake(count, 1, 1);

        [encoder dispatchThreads:grid threadsPerThreadgroup:threadsPerThreadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (*(unsigned int*)[foundBuffer contents] != 0) {
            memcpy(result, [resultBuffer contents], 8); // 8 bytes for result
            return 1;
        }
        return 0;
    }
}

// 清理资源
void cleanupMetal() {
    device = nil;
    commandQueue = nil;
    computePipelineState = nil;
    resultBuffer = nil;
    targetBuffer = nil;
    foundBuffer = nil;
    baseIndexBuffer = nil;
}
*/
import "C"

import (
	"context"
	"encoding/hex"
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/schollz/progressbar/v3"
)

var (
	GPUCores                 int
	MaxThreadsPerThreadgroup int
	GPUBatchSize             uint64
)

var (
	// sha1(b"deadbeef").digest().hex() -> f49cf6381e322b147053b74e4500af8533ac1e4c
	tarHex   = "f49cf6381e322b147053b74e4500af8533ac1e4c"
	tarBytes []byte
)

var (
	globalSearchIndex atomic.Uint64 // Global counter for the search space
	foundFlag         atomic.Int32
	foundResult       [8]byte // Result is an 8-char string
	resultMutex       sync.Mutex
)

func init() {
	var err error
	tarBytes, err = hex.DecodeString(tarHex)
	if err != nil {
		log.Fatalf("无法解码目标哈希: %v", err)
	}

	fmt.Println("初始化 Metal GPU...")
	var gpuInfo C.GPUInfo
	if ret := C.initMetal(&gpuInfo); ret != 0 {
		log.Fatalf("Metal 初始化失败")
	}
	GPUCores = int(gpuInfo.coreCount)
	MaxThreadsPerThreadgroup = int(gpuInfo.maxThreadsPerThreadgroup)
	GPUBatchSize = uint64(GPUCores * MaxThreadsPerThreadgroup * 64) // 64x oversubscription
	if GPUBatchSize > (1 << 24) {                                   // Max 16M hashes per batch
		GPUBatchSize = 1 << 24
	}

	fmt.Printf("\n=== Metal 配置 ===\n")
	fmt.Printf("GPU核心数: %d\n", GPUCores)
	fmt.Printf("最大线程组大小: %d\n", MaxThreadsPerThreadgroup)
	fmt.Printf("批处理大小: %d (%.2fM hashes)\n", GPUBatchSize, float64(GPUBatchSize)/(1024*1024))
	fmt.Println("\nMetal GPU 初始化成功！")
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Total search space is 16^8
	totalOperations := int64(math.Pow(16, 8)) // 2^32
	bar := progressbar.NewOptions64(totalOperations,
		progressbar.OptionSetDescription(fmt.Sprintf("SHA-1 on %d-core GPU...", GPUCores)),
		progressbar.OptionShowBytes(false),
		progressbar.OptionSetWidth(30),
		progressbar.OptionShowCount(),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer: "█", SaucerHead: "█", SaucerPadding: " ",
			BarStart: "[", BarEnd: "]",
		}),
		progressbar.OptionThrottle(100*time.Millisecond),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("H/s"),
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	wg := &sync.WaitGroup{}

	numSchedulers := runtime.NumCPU()
	if numSchedulers > 4 {
		numSchedulers = 4
	}

	timeStart := time.Now()

	fmt.Printf("\n开始使用 %d 个 Go 调度器在 GPU 上爆破 SHA-1...\n", numSchedulers)
	fmt.Printf("总搜索空间: %d (16^8)\n\n", totalOperations)

	for i := 0; i < numSchedulers; i++ {
		wg.Add(1)
		go gpuScheduler(i, wg, ctx, cancel, uint64(totalOperations), bar)
	}

	wg.Wait()

	C.cleanupMetal()

	timeEnd := time.Now()
	if foundFlag.Load() != 0 {
		bar.Finish()
	} else {
		bar.Set64(totalOperations)
	}

	duration := timeEnd.Sub(timeStart)
	totalHashes := globalSearchIndex.Load()
	if totalHashes > uint64(totalOperations) {
		totalHashes = uint64(totalOperations)
	}

	hashesPerSecond := float64(totalHashes) / duration.Seconds()

	fmt.Printf("\n\n=== GPU 性能统计 ===\n")
	fmt.Printf("GPU: %d核\n", GPUCores)
	fmt.Printf("总耗时: %v\n", duration.Round(time.Millisecond))
	fmt.Printf("总哈希数: %d\n", totalHashes)
	fmt.Printf("哈希速率: %.2f MH/s\n", hashesPerSecond/1000000)
	fmt.Printf("每核心速率: %.2f MH/s\n", hashesPerSecond/1000000/float64(GPUCores))

	if foundFlag.Load() != 0 {
		resultMutex.Lock()
		resultStr := string(foundResult[:])
		resultMutex.Unlock()
		fmt.Printf("\n🎉 找到结果: %s\n", resultStr)
		fmt.Printf("   对应哈希: %s\n", tarHex)
	} else {
		fmt.Printf("\n未找到结果。\n")
	}
}

func gpuScheduler(id int, wg *sync.WaitGroup, ctx context.Context, cancel context.CancelFunc, totalOps uint64, bar *progressbar.ProgressBar) {
	defer wg.Done()

	result := make([]byte, 8)
	cTarget := (*C.uint8_t)(unsafe.Pointer(&tarBytes[0]))
	cResult := (*C.uint8_t)(unsafe.Pointer(&result[0]))
	cMaxThreads := C.int(MaxThreadsPerThreadgroup)

	for {
		// 如果已找到或上下文已取消，则退出
		if foundFlag.Load() != 0 {
			return
		}
		select {
		case <-ctx.Done():
			return
		default:
		}

		startIndex := globalSearchIndex.Add(GPUBatchSize) - GPUBatchSize
		if startIndex >= totalOps {
			return
		}

		batchSize := GPUBatchSize
		if startIndex+batchSize > totalOps {
			batchSize = totalOps - startIndex
		}

		ret := C.searchOnGPU(
			C.uint64_t(startIndex),
			C.uint64_t(batchSize),
			cTarget,
			cResult,
			cMaxThreads,
		)

		bar.Add64(int64(batchSize))

		if ret == 1 {
			if foundFlag.CompareAndSwap(0, 1) {
				// 确保只设置一次结果并取消其他任务
				resultMutex.Lock()
				copy(foundResult[:], result)
				resultMutex.Unlock()
				fmt.Printf("\n[GPU Scheduler %d] 发现匹配项! 正在停止其他任务...\n", id)
				cancel()
			}
			return
		}
	}
}
