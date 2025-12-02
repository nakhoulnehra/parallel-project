// cuda_sla_storms.cu — CUDA version of SLA violation + storm detection
// Same logic & results as CPU versions. Loads ONLY data-2025-10.csv.
//
// GPU work: for every violating complaint i, count prior violations j
// in the SAME ZIP within the 7-day window. We tile j into shared memory.
//
// CUDA practices applied here (as applicable to this workload):
//  - Kernel launch config (tunable block size, grid-stride loop).
//  - Shared memory tiling of j-side data (times, ZIPs, flags).
//  - Memory coalescing via Structure-of-Arrays layout on device.
//  - Local caching of i's ZIP into registers to reduce global access.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <glob.h>
#include <cuda_runtime.h>

#define MAX_LINE 2048
#define MAX_COMPLAINTS 2000000
#define ROLLING_WINDOW_DAYS 7
#define ZIP_CHARS 16         // incident_zip stored as fixed 16 chars

// ----------------------- Host-side structures -----------------------
typedef struct {
    char    unique_key[32];
    time_t  created_date;
    time_t  closed_date;
    double  response_hours;
    char    status[16];
    char    complaint_type[128];
    char    borough[32];
    char    incident_zip[ZIP_CHARS];
    int     is_weekend;
} Complaint;

// ----------------------- CUDA helpers -------------------------------
#define CUDA_TRY(call) do {                                           \
    cudaError_t _err = (call);                                        \
    if (_err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ----------------------- Parsing & SLA (host) -----------------------
static inline time_t parse_date(const char* date_str) {
    if (strcmp(date_str, "") == 0) return 0;
    struct tm tmv = {0};
    // 2025-05-25T05:14:07.000Z
    sscanf(date_str, "%d-%d-%dT%d:%d:%d",
           &tmv.tm_year, &tmv.tm_mon, &tmv.tm_mday,
           &tmv.tm_hour, &tmv.tm_min, &tmv.tm_sec);
    tmv.tm_year -= 1900;
    tmv.tm_mon  -= 1;
    tmv.tm_isdst = -1;
    return mktime(&tmv);
}

static inline double get_sla_threshold(const char* complaint_type, const char* borough, int is_weekend) {
    double base_hours = 72.0;
    if (strstr(complaint_type, "Food Poisoning") ||
        strstr(complaint_type, "Unsanitary") ||
        strstr(complaint_type, "Rodent")) {
        base_hours = 24.0;
    } else if (strstr(complaint_type, "Noise") ||
               strstr(complaint_type, "Heat/Hot Water")) {
        base_hours = 48.0;
    }
    if (strcmp(borough, "MANHATTAN") == 0 || strcmp(borough, "BROOKLYN") == 0) {
        base_hours *= 0.8;
    }
    if (is_weekend) {
        base_hours *= 1.5;
    }
    return base_hours;
}

// CSV loader (header-driven positions consistent with your prior code)
static int load_csv_file(const char* filename, Complaint* complaints, int* count) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Warning: Cannot open file %s\n", filename);
        return 0;
    }
    char line[MAX_LINE];
    if (fgets(line, MAX_LINE, fp) == NULL) { fclose(fp); return 0; }

    int loaded = 0;
    while (fgets(line, MAX_LINE, fp) && *count < MAX_COMPLAINTS) {
        char* token; char* saveptr; int col = 0;
        token = strtok_r(line, ",", &saveptr);
        while (token) {
            switch(col) {
                case 0: strncpy(complaints[*count].unique_key, token, 31); complaints[*count].unique_key[31]='\0'; break;
                case 1: complaints[*count].created_date = parse_date(token); break;
                case 2: complaints[*count].closed_date  = parse_date(token); break;
                case 3: complaints[*count].response_hours = atof(token); break;
                case 4: strncpy(complaints[*count].status, token, 15); complaints[*count].status[15]='\0'; break;
                case 7: strncpy(complaints[*count].complaint_type, token, 127); complaints[*count].complaint_type[127]='\0'; break;
                case 9: strncpy(complaints[*count].borough, token, 31); complaints[*count].borough[31]='\0'; break;
                case 11:{
                    // keep exactly as read (to match CPU equality semantics)
                    size_t L = strnlen(token, ZIP_CHARS-1);
                    memset(complaints[*count].incident_zip, 0, ZIP_CHARS);
                    strncpy(complaints[*count].incident_zip, token, ZIP_CHARS-1);
                } break;
                case 19: complaints[*count].is_weekend = atoi(token); break;
            }
            token = strtok_r(NULL, ",", &saveptr);
            col++;
        }
        (*count)++; loaded++;
    }
    fclose(fp);
    return loaded;
}

// ----------------------- GPU kernel -------------------------------
// Inputs:
//  times_sec:  N x int64 (created timestamps, seconds)
//  zips:       N x ZIP_CHARS bytes (flattened)
//  isViol:     N x uint8 (1 if response_hours > SLA)
// Output:
//  outCounts:  N x int (rolling count for each i)
__global__ void rolling_count_kernel(const long long* __restrict__ times_sec,
                                     const unsigned char* __restrict__ isViol,
                                     const char* __restrict__ zips,
                                     int N,
                                     int* __restrict__ outCounts)
{
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;
    const int gdim = gridDim.x;
    int i = blockIdx.x * bdim + tid;

    // Shared memory tiling for j-side:
    extern __shared__ unsigned char s_mem[];
    // Layout: [times | isV | zips]
    long long*  s_times = (long long*)s_mem;
    unsigned char* s_v  = (unsigned char*)(s_times + bdim);
    char* s_zips        = (char*)(s_v + bdim); // size bdim * ZIP_CHARS

    const double DAY = 86400.0;

    // Grid-stride over i
    for (; i < N; i += bdim * gdim) {

        if (!isViol[i]) { // Only compute for violating i, exact CPU parity
            outCounts[i] = 0;
            continue;
        }

        const long long ti = times_sec[i];

        // Cache i's ZIP locally (registers / local memory)
        char zi[ZIP_CHARS];
        #pragma unroll
        for (int c = 0; c < ZIP_CHARS; ++c)
            zi[c] = zips[i * ZIP_CHARS + c];

        int cnt = 0;

        // Tile over j in chunks of blockDim.x
        for (int base = 0; base < N; base += bdim) {
            int j = base + tid;

            // Coalesced loads of the tile into shared memory
            if (j < N) {
                s_times[tid] = times_sec[j];
                s_v[tid]     = isViol[j];
                // store ZIP_CHARS bytes for this j
                #pragma unroll
                for (int c = 0; c < ZIP_CHARS; ++c)
                    s_zips[tid * ZIP_CHARS + c] = zips[j * ZIP_CHARS + c];
            } else {
                s_v[tid] = 0;    // mark as non-violation/out-of-range
            }

            __syncthreads();

            const int tileCount = min(bdim, N - base);

            // Consume the tile
            for (int k = 0; k < tileCount; ++k) {
                const int global_j = base + k;
                if (global_j == i) continue;      // skip self
                if (!s_v[k])        continue;      // count only if j itself is a violation

                // compare ZIPs (exact byte-wise)
                bool same_zip = true;
                #pragma unroll
                for (int c = 0; c < ZIP_CHARS; ++c) {
                    if (zi[c] != s_zips[k * ZIP_CHARS + c]) { same_zip = false; break; }
                }
                if (!same_zip) continue;

                // time window: j must be within 7 days BEFORE i (>=0 && <=7)
                double days = double(ti - s_times[k]) / DAY;
                if (days >= 0.0 && days <= double(ROLLING_WINDOW_DAYS)) {
                    cnt++;
                }
            }

            __syncthreads();
        }

        outCounts[i] = cnt;
    }
}

// ----------------------- Main (host) -------------------------------
int main(int argc, char* argv[]) {
    int BLOCK = 256;                    // default block size (tunable)
    if (argc >= 2) {
        int v = atoi(argv[1]);
        if (v > 0) BLOCK = v;
    }

    double t0 = (double)clock() / CLOCKS_PER_SEC;

    Complaint* complaints = (Complaint*)malloc(MAX_COMPLAINTS * sizeof(Complaint));
    if (!complaints) { fprintf(stderr, "Allocation failed\n"); return 1; }

    int count = 0;
    printf("=== LOADING DATA ===\n");
    glob_t g;
    // Final decision: ONLY October (month 10)
    glob("data-2025-10.csv", 0, NULL, &g);

    for (size_t i = 0; i < g.gl_pathc; i++) {
        printf("Loading %s...\n", g.gl_pathv[i]);
        int loaded = load_csv_file(g.gl_pathv[i], complaints, &count);
        printf("  Loaded %d (Total: %d)\n", loaded, count);
    }
    globfree(&g);

    if (count == 0) {
        printf("Error: No data loaded\n");
        free(complaints);
        return 1;
    }

    // Precompute SLA & violation flags on host (keeps GPU pure numeric)
    unsigned char* h_isViol = (unsigned char*)malloc(count * sizeof(unsigned char));
    long long*     h_times  = (long long*)    malloc(count * sizeof(long long));
    char*          h_zips   = (char*)         malloc((size_t)count * ZIP_CHARS);

    if (!h_isViol || !h_times || !h_zips) {
        fprintf(stderr, "Host arrays allocation failed\n");
        return 1;
    }

    int total_violations = 0;
    for (int i = 0; i < count; i++) {
        double sla = get_sla_threshold(complaints[i].complaint_type,
                                       complaints[i].borough,
                                       complaints[i].is_weekend);
        const int viol = (complaints[i].response_hours > sla) ? 1 : 0;
        h_isViol[i] = (unsigned char)viol;
        h_times[i]  = (long long)complaints[i].created_date;
        // copy fixed ZIP_CHARS bytes as-is for exact equality semantics
        #pragma unroll
        for (int c = 0; c < ZIP_CHARS; ++c)
            h_zips[i * ZIP_CHARS + c] = complaints[i].incident_zip[c];

        total_violations += viol;
    }

    double t1 = (double)clock() / CLOCKS_PER_SEC;
    printf("\nLoaded %d complaints (month 10 only)\n", count);
    printf("Host precompute (SLA & flags) time: %.2f s\n", (t1 - t0));

    // -------------------- Device allocations --------------------
    long long*     d_times = nullptr;
    unsigned char* d_isViol = nullptr;
    char*          d_zips  = nullptr;
    int*           d_counts = nullptr;

    CUDA_TRY(cudaMalloc(&d_times,  count * sizeof(long long)));
    CUDA_TRY(cudaMalloc(&d_isViol, count * sizeof(unsigned char)));
    CUDA_TRY(cudaMalloc(&d_zips,   (size_t)count * ZIP_CHARS));
    CUDA_TRY(cudaMalloc(&d_counts, count * sizeof(int)));

    CUDA_TRY(cudaMemcpy(d_times,  h_times,  count * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_isViol, h_isViol, count * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_zips,   h_zips,   (size_t)count * ZIP_CHARS,    cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemset(d_counts, 0,        count * sizeof(int)));

    // -------------------- Launch configuration --------------------
    int grid = (count + BLOCK - 1) / BLOCK;
    // avoid over-large grids; enough to saturate GPU
    grid = (grid > 32768) ? 32768 : grid;

    size_t shmem_bytes = BLOCK * (sizeof(long long) + sizeof(unsigned char) + ZIP_CHARS * sizeof(char));

    printf("\n=== CUDA KERNEL ===\n");
    printf("Grid: %d  Block: %d  SharedMem: %.1f KB per block\n",
           grid, BLOCK, shmem_bytes / 1024.0);

    cudaEvent_t evStart, evStop;
    CUDA_TRY(cudaEventCreate(&evStart));
    CUDA_TRY(cudaEventCreate(&evStop));
    CUDA_TRY(cudaEventRecord(evStart));

    rolling_count_kernel<<<grid, BLOCK, shmem_bytes>>>(d_times, d_isViol, d_zips, count, d_counts);
    CUDA_TRY(cudaPeekAtLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    CUDA_TRY(cudaEventRecord(evStop));
    CUDA_TRY(cudaEventSynchronize(evStop));
    float kernel_ms = 0.f;
    CUDA_TRY(cudaEventElapsedTime(&kernel_ms, evStart, evStop));

    // -------------------- Copy back & finalize on host --------------------
    int* h_counts = (int*)malloc(count * sizeof(int));
    CUDA_TRY(cudaMemcpy(h_counts, d_counts, count * sizeof(int), cudaMemcpyDeviceToHost));

    int storm_complaints = 0;
    for (int i = 0; i < count; i++) {
        if (h_isViol[i] && h_counts[i] >= 5) storm_complaints++;
    }

    double t2 = (double)clock() / CLOCKS_PER_SEC;

    printf("\n=== RESULTS (CUDA) ===\n");
    printf("Total complaints: %d\n", count);
    printf("SLA violations: %d (%.2f%%)\n",
           total_violations, (double)total_violations / (double)count * 100.0);
    printf("Complaint storms (5+ violations in 7-day window): %d\n", storm_complaints);

    printf("\n=== TIMING ===\n");
    printf("Host load + precompute: %.2f s\n", (t1 - t0));
    printf("Kernel time: %.3f ms\n", kernel_ms);
    printf("Total wall time (host perspective): %.2f s\n", (t2 - t0));

    // -------------------- Cleanup --------------------
    free(complaints);
    free(h_isViol); free(h_times); free(h_zips); free(h_counts);
    CUDA_TRY(cudaFree(d_times));
    CUDA_TRY(cudaFree(d_isViol));
    CUDA_TRY(cudaFree(d_zips));
    CUDA_TRY(cudaFree(d_counts));
    CUDA_TRY(cudaEventDestroy(evStart));
    CUDA_TRY(cudaEventDestroy(evStop));

    return 0;
}
