#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <glob.h>
#include <cuda_runtime.h>

#define MAX_LINE 2048
#define MAX_COMPLAINTS 2000000
#define ROLLING_WINDOW_DAYS 7

typedef struct {
    char unique_key[32];
    time_t created_date;
    time_t closed_date;
    double response_hours;
    char status[16];
    char complaint_type[128];
    char borough[32];
    char incident_zip[16];
    int is_weekend;
} Complaint;

#define CUDA_TRY(call) do {                                           \
    cudaError_t _err = (call);                                        \
    if (_err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)


// Parse ISO 8601 date format: 2025-05-25T05:14:07.000Z
static time_t parse_date(const char* date_str) {
    if (strcmp(date_str, "") == 0) return 0;

    struct tm tmv = {0};
    sscanf(date_str, "%d-%d-%dT%d:%d:%d",
           &tmv.tm_year, &tmv.tm_mon, &tmv.tm_mday,
           &tmv.tm_hour, &tmv.tm_min, &tmv.tm_sec);

    tmv.tm_year -= 1900;
    tmv.tm_mon  -= 1;
    tmv.tm_isdst = -1;

    return mktime(&tmv);
}

// Same as in your OpenMP version
static double get_sla_threshold(const char* complaint_type,
                                const char* borough,
                                int is_weekend) {
    double base_hours = 72.0; // Default 3 days

    // Urgent complaint types
    if (strstr(complaint_type, "Food Poisoning") ||
        strstr(complaint_type, "Unsanitary") ||
        strstr(complaint_type, "Rodent")) {
        base_hours = 24.0;
    } else if (strstr(complaint_type, "Noise") ||
               strstr(complaint_type, "Heat/Hot Water")) {
        base_hours = 48.0;
    }

    // Manhattan and Brooklyn have stricter SLAs
    if (strcmp(borough, "MANHATTAN") == 0 ||
        strcmp(borough, "BROOKLYN") == 0) {
        base_hours *= 0.8;
    }

    // Weekend complaints get extended SLA
    if (is_weekend) {
        base_hours *= 1.5;
    }

    return base_hours;
}

// CSV loader: same mapping as your OpenMP version,
// with explicit null-termination for safety.
static int load_csv_file(const char* filename, Complaint* complaints, int* count) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Warning: Cannot open file %s\n", filename);
        return 0;
    }

    char line[MAX_LINE];

    // Skip header
    if (fgets(line, MAX_LINE, fp) == NULL) {
        fclose(fp);
        return 0;
    }

    int loaded = 0;
    while (fgets(line, MAX_LINE, fp) && *count < MAX_COMPLAINTS) {
        char* token;
        char* saveptr;
        int col = 0;

        token = strtok_r(line, ",", &saveptr);
        while (token && col < 20) {
            switch(col) {
                case 0:
                    strncpy(complaints[*count].unique_key, token, 31);
                    complaints[*count].unique_key[31] = '\0';
                    break;
                case 1:
                    complaints[*count].created_date = parse_date(token);
                    break;
                case 2:
                    complaints[*count].closed_date = parse_date(token);
                    break;
                case 3:
                    complaints[*count].response_hours = atof(token);
                    break;
                case 4:
                    strncpy(complaints[*count].status, token, 15);
                    complaints[*count].status[15] = '\0';
                    break;
                case 7:
                    strncpy(complaints[*count].complaint_type, token, 127);
                    complaints[*count].complaint_type[127] = '\0';
                    break;
                case 9:
                    strncpy(complaints[*count].borough, token, 31);
                    complaints[*count].borough[31] = '\0';
                    break;
                case 11:
                    strncpy(complaints[*count].incident_zip, token, 15);
                    complaints[*count].incident_zip[15] = '\0';
                    break;
                case 19:
                    complaints[*count].is_weekend = atoi(token);
                    break;
            }
            token = strtok_r(NULL, ",", &saveptr);
            col++;
        }

        (*count)++;
        loaded++;
    }

    fclose(fp);
    return loaded;
}

__device__ bool same_zip_16(const char* a, const char* b) {
    for (int c = 0; c < 16; ++c) {
        unsigned char ac = (unsigned char)a[c];
        unsigned char bc = (unsigned char)b[c];
        if (ac != bc) return false;
        if (ac == '\0' && bc == '\0') return true;
    }
    // If we exit the loop without returning, they are equal across all 16.
    return true;
}

__global__ void rolling_count_kernel(const Complaint* complaints,
                                     const unsigned char* isViol,
                                     int N,
                                     int* outCounts)
{
    const int tid    = threadIdx.x;
    const int bdim   = blockDim.x;
    const int gdim   = gridDim.x;
    int i = blockIdx.x * bdim + tid;

    const double DAY = 24.0 * 3600.0;

    // Grid-stride loop over i
    for (; i < N; i += bdim * gdim) {

        if (!isViol[i]) {
            outCounts[i] = 0;
            continue;
        }

        const time_t current_time = complaints[i].created_date;
        char current_zip[16];

        // Cache ZIP of i locally
        for (int c = 0; c < 16; ++c) {
            current_zip[c] = complaints[i].incident_zip[c];
        }

        int cnt = 0;

        // Full scan over all j, same as CPU logic
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;

            if (!isViol[j]) continue;

            // Same ZIP?
            if (!same_zip_16(current_zip, complaints[j].incident_zip))
                continue;

            time_t other_time = complaints[j].created_date;
            double days_diff =
                double(current_time - other_time) / DAY;

            if (days_diff >= 0.0 &&
                days_diff <= double(ROLLING_WINDOW_DAYS)) {
                cnt++;
            }
        }

        outCounts[i] = cnt;
    }
}

int main(int argc, char* argv[]) {
    int BLOCK = 256;  // tunable block size
    if (argc >= 2) {
        int v = atoi(argv[1]);
        if (v > 0) BLOCK = v;
    }

    double t0 = (double)clock() / CLOCKS_PER_SEC;

    Complaint* complaints =
        (Complaint*)malloc(MAX_COMPLAINTS * sizeof(Complaint));
    if (!complaints) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }

    int count = 0;

    printf("=== LOADING DATA ===\n");

    glob_t glob_result;
    glob("data-2025-10.csv", GLOB_TILDE, NULL, &glob_result);

    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        printf("Loading %s...\n", glob_result.gl_pathv[i]);
        int loaded = load_csv_file(glob_result.gl_pathv[i],
                                   complaints, &count);
        printf("  Loaded %d records (Total: %d)\n", loaded, count);
    }

    globfree(&glob_result);

    if (count == 0) {
        printf("Error: No data loaded\n");
        free(complaints);
        return 1;
    }

    unsigned char* h_isViol =
        (unsigned char*)malloc(count * sizeof(unsigned char));
    if (!h_isViol) {
        fprintf(stderr, "Error: isViol allocation failed\n");
        free(complaints);
        return 1;
    }

    int total_violations = 0;
    for (int i = 0; i < count; ++i) {
        double sla = get_sla_threshold(complaints[i].complaint_type,
                                       complaints[i].borough,
                                       complaints[i].is_weekend);
        int viol = (complaints[i].response_hours > sla) ? 1 : 0;
        h_isViol[i] = (unsigned char)viol;
        total_violations += viol;
    }

    double t1 = (double)clock() / CLOCKS_PER_SEC;
    printf("\nLoaded %d complaints (month 10 only)\n", count);
    printf("Host precompute (SLA & flags) time: %.2f s\n", (t1 - t0));

    Complaint*    d_complaints = nullptr;
    unsigned char* d_isViol    = nullptr;
    int*          d_counts     = nullptr;

    CUDA_TRY(cudaMalloc(&d_complaints,
                        count * sizeof(Complaint)));
    CUDA_TRY(cudaMalloc(&d_isViol,
                        count * sizeof(unsigned char)));
    CUDA_TRY(cudaMalloc(&d_counts,
                        count * sizeof(int)));

    CUDA_TRY(cudaMemcpy(d_complaints, complaints,
                        count * sizeof(Complaint),
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_isViol, h_isViol,
                        count * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemset(d_counts, 0,
                        count * sizeof(int)));

    int grid = (count + BLOCK - 1) / BLOCK;
    if (grid > 32768) grid = 32768;

    printf("\n=== CUDA KERNEL ===\n");
    printf("Grid: %d  Block: %d\n", grid, BLOCK);

    cudaEvent_t evStart, evStop;
    CUDA_TRY(cudaEventCreate(&evStart));
    CUDA_TRY(cudaEventCreate(&evStop));
    CUDA_TRY(cudaEventRecord(evStart));

    rolling_count_kernel<<<grid, BLOCK>>>(d_complaints,
                                          d_isViol,
                                          count,
                                          d_counts);
    CUDA_TRY(cudaPeekAtLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    CUDA_TRY(cudaEventRecord(evStop));
    CUDA_TRY(cudaEventSynchronize(evStop));
    float kernel_ms = 0.0f;
    CUDA_TRY(cudaEventElapsedTime(&kernel_ms,
                                  evStart, evStop));

    int* h_counts = (int*)malloc(count * sizeof(int));
    if (!h_counts) {
        fprintf(stderr, "Error: h_counts allocation failed\n");
        free(complaints);
        free(h_isViol);
        CUDA_TRY(cudaFree(d_complaints));
        CUDA_TRY(cudaFree(d_isViol));
        CUDA_TRY(cudaFree(d_counts));
        return 1;
    }

    CUDA_TRY(cudaMemcpy(h_counts, d_counts,
                        count * sizeof(int),
                        cudaMemcpyDeviceToHost));

    int storm_complaints = 0;
    for (int i = 0; i < count; ++i) {
        if (h_isViol[i] && h_counts[i] >= 5) {
            storm_complaints++;
        }
    }

    double t2 = (double)clock() / CLOCKS_PER_SEC;

    printf("\n=== RESULTS (CUDA) ===\n");
    printf("Total complaints: %d\n", count);
    printf("SLA violations: %d (%.2f%%)\n",
           total_violations,
           (double)total_violations / (double)count * 100.0);
    printf("Complaint storms (5+ violations in 7-day window): %d\n",
           storm_complaints);

    printf("\n=== TIMING ===\n");
    printf("Host load + precompute: %.2f s\n", (t1 - t0));
    printf("Kernel time: %.3f ms\n", kernel_ms);
    printf("Total wall time (host perspective): %.2f s\n",
           (t2 - t0));

    // free memory
    free(complaints);
    free(h_isViol);
    free(h_counts);

    CUDA_TRY(cudaFree(d_complaints));
    CUDA_TRY(cudaFree(d_isViol));
    CUDA_TRY(cudaFree(d_counts));
    CUDA_TRY(cudaEventDestroy(evStart));
    CUDA_TRY(cudaEventDestroy(evStop));

    return 0;
}
