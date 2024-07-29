#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                      // a)
    asm("fence.proxy.async.shared::cta;");
    // ptx::fence_proxy_async(ptx::space_shared);   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
    cuda::memcpy_async(
        smem_data, 
        data + offset, 
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar
    );
  }
  // 3b. All threads arrive on the barrier
  barrier::arrival_token token = bar.arrive();
  
  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  asm("fence.proxy.async.shared::cta;");
  // ptx::fence_proxy_async(ptx::space_shared);   // b)
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    // This comes from cute's `cast_smem_ptr_to_uint`
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_data));
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
       :: "l"(data + offset), "r"(smem_int_ptr), "n"(sizeof(smem_data))
       );
    /* ptx::cp_async_bulk(
        ptx::space_global,
        ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data)); */
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    asm("cp.async.bulk.commit_group;");
    // ptx::cp_async_bulk_commit_group();

    // Wait for the group to have completed reading from shared memory.
    
    asm("cp.async.bulk.wait_group.read 0;");
    // ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}
