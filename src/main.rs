use std::{ptr, thread, sync::Arc};
use std::ffi::c_void;

type CUdeviceptr = *mut u8;

// FFI bindings to call CUDA API with Rust code
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    // pinned memory for higher bandwidth
    fn cudaMalloc(ptr: *mut CUdeviceptr, size: usize) -> i32;
    fn cudaFree(ptr: CUdeviceptr) -> i32;
    // async
    fn cudaMemcpyAsync(
        dst: CUdeviceptr,
        src: *const u8,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;

    // blocks CPU thread until all CUDA ops are completed
    fn cudaDeviceSynchronize() -> i32;

    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;

    fn cudaDeviceEnablePeerAccess(peerDevice: i32, flags: u32) -> i32;
    // GPU to GPU data copy
    fn cudaMemcpyPeer(
        dst: CUdeviceptr,
        dstDevice: i32,
        src: CUdeviceptr,
        srcDevice: i32,
        count: usize,
    ) -> i32;

    // functions to time the cuda events for benchmarking
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, stop: *mut c_void) -> i32;
}

const H2D: i32 = 1;
const CUDA_SUCCESS: i32 = 0;

// Wrapper for pinned host memory pointer
struct HostPtr(*mut u8);
unsafe impl Send for HostPtr {}
unsafe impl Sync for HostPtr {}

// wrapper for device pointer
struct DevicePtr(CUdeviceptr);
unsafe impl Send for DevicePtr {}

fn main() {
    println!("Below is the test with 2 GPUs, total 0.50 GB");

    unsafe {
        let size: usize = 256 * 1024 * 1024; // 256 MB

        // allocated pinned host memory
        let mut host_ptr_raw: *mut c_void = ptr::null_mut();
        assert_eq!(cudaHostAlloc(&mut host_ptr_raw, size, 0), CUDA_SUCCESS);
        let host_ptr = Arc::new(HostPtr(host_ptr_raw as *mut u8));

        // initialize memory
        let slice = std::slice::from_raw_parts_mut(host_ptr.0, size);
        slice.fill(1);

        // spawn two threads that transfer CPU to GPU at the same time

        let t0 = {
            let host_clone = Arc::clone(&host_ptr);
            thread::spawn(move || unsafe {
                cudaSetDevice(0);

                // Allocate device memory
                let mut d0: CUdeviceptr = ptr::null_mut();
                cudaMalloc(&mut d0, size);

                // Create CUDA events
                let mut start: *mut c_void = ptr::null_mut();
                let mut stop: *mut c_void = ptr::null_mut();
                cudaEventCreate(&mut start);
                cudaEventCreate(&mut stop);

                // Record start
                cudaEventRecord(start, ptr::null_mut());

                // Launch asynchronous H2D copy
                cudaMemcpyAsync(d0, host_clone.0, size, H2D, ptr::null_mut());

                // Record stop
                cudaEventRecord(stop, ptr::null_mut());

                // Wait for the copy to finish
                cudaEventSynchronize(stop);

                // Measure elapsed time
                let mut ms: f32 = 0.0;
                cudaEventElapsedTime(&mut ms, start, stop);

                println!("Thread 0 H2D transfer time: {:.3} ms", ms);

                DevicePtr(d0)
            })
        };

        let t1 = {
            let host_clone = Arc::clone(&host_ptr);
            thread::spawn(move || unsafe {
                cudaSetDevice(1);
                let mut d1: CUdeviceptr = ptr::null_mut();
                cudaMalloc(&mut d1, size);

                let mut start: *mut c_void = ptr::null_mut();
                let mut stop: *mut c_void = ptr::null_mut();
                cudaEventCreate(&mut start);
                cudaEventCreate(&mut stop);

                cudaEventRecord(start, ptr::null_mut());

                // transfer data
                cudaMemcpyAsync(d1, host_clone.0, size, H2D, ptr::null_mut());

                cudaEventRecord(stop, ptr::null_mut());
                cudaEventSynchronize(stop);

                let mut ms: f32 = 0.0;
                cudaEventElapsedTime(&mut ms, start, stop);
                println!("Thread 1 H2D transfer time: {:.3} ms", ms);

                DevicePtr(d1)
            })
        };

        // waits for threads to finish and extract inner value
        let DevicePtr(d0) = t0.join().unwrap();
        let DevicePtr(d1) = t1.join().unwrap();

        // enable transfer between the two GPUs
        cudaSetDevice(0);
        let _ = cudaDeviceEnablePeerAccess(1, 0);
        cudaSetDevice(1);
        let _ = cudaDeviceEnablePeerAccess(0, 0);

        // create CUDA events
        cudaSetDevice(0);
        let mut start: *mut c_void = ptr::null_mut();
        let mut stop: *mut c_void = ptr::null_mut();
        cudaEventCreate(&mut start);
        cudaEventCreate(&mut stop);

        cudaEventRecord(start, ptr::null_mut());
        // transfer from GPU 1 to GPU 0
        cudaMemcpyPeer(d0, 0, d1, 1, size);
        cudaEventRecord(stop, ptr::null_mut());
        cudaEventSynchronize(stop);

        let mut ms: f32 = 0.0;
        cudaEventElapsedTime(&mut ms, start, stop);

        let gb = size as f32 / (1024.0 * 1024.0 * 1024.0);
        let bandwidth = gb / (ms / 1000.0);

        println!("P2P GPU1 -> GPU0 transfer: {:.3} GB", gb);
        println!("Time: {:.3} ms", ms);
        println!("Peer to peer bandwidth: {:.3} GB/s", bandwidth);

        // free up the memory 
        cudaFree(d1);
        cudaFreeHost(host_ptr.0 as *mut c_void);

        //////////////////////////////////////////////////
        // now testing with 1 GPU, regular solution:
        println!("\nBelow is the test with 1 GPU, 0.50 GB");

        let size = 512 * 1024 * 1024; // 512 MB

        // allocated pinned host memory
        let mut host_ptr_raw: *mut c_void = ptr::null_mut();
        assert_eq!(cudaHostAlloc(&mut host_ptr_raw, size, 0), CUDA_SUCCESS);
        let host_ptr = Arc::new(HostPtr(host_ptr_raw as *mut u8));

        // initialize memory
        let slice = std::slice::from_raw_parts_mut(host_ptr.0, size);
        slice.fill(1);

        // set the GPU
        cudaSetDevice(0);
        let mut d0: CUdeviceptr = ptr::null_mut();
        cudaMalloc(&mut d0, size);

        // create events to time
        let mut start: *mut c_void = ptr::null_mut();
        let mut stop: *mut c_void = ptr::null_mut();
        cudaEventCreate(&mut start);
        cudaEventCreate(&mut stop);

        // start recording
        cudaEventRecord(start, ptr::null_mut());

        // perform the transfer
        cudaMemcpyAsync(d0, host_ptr.0, size, H2D, ptr::null_mut());

        // record stop
        cudaEventRecord(stop, ptr::null_mut());
    
        cudaEventSynchronize(stop);

        let mut ms: f32 = 0.0;
        cudaEventElapsedTime(&mut ms, start, stop);

        let gb = size as f32 / (1024.0 * 1024.0 * 1024.0);
        let bandwidth = gb / (ms / 1000.0);

        println!("H2D transfer time: {:.3} GB", gb);
        println!("Time: {:.3} ms", ms);
        println!("Bandwidth for PCIe bus: {:.3} GB/s", bandwidth);
    }
}