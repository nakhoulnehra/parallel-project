// pthread_sla_storms.c — Pthreads version of SLA violation + storm detection
// Same logic as sequential/MPI versions, but parallelized with POSIX threads.
// Uses only data-2025-09.csv and data-2025-10.csv.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <glob.h>
#include <pthread.h>
#include <sys/time.h>

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

typedef struct {
    Complaint *complaints;
    int total;
    int start_idx;
    int end_idx;
    int local_violations;
    int local_storms;
} ThreadArgs;

// Simple wall-clock timing helper
double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

// Parse ISO 8601 date format: 2025-05-25T05:14:07.000Z
time_t parse_date(const char* date_str) {
    if (strcmp(date_str, "") == 0) return 0;

    struct tm tm = {0};
    sscanf(date_str, "%d-%d-%dT%d:%d:%d",
           &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
           &tm.tm_hour, &tm.tm_min, &tm.tm_sec);

    tm.tm_year -= 1900;
    tm.tm_mon  -= 1;
    tm.tm_isdst = -1;

    return mktime(&tm);
}

// Get SLA threshold in hours based on complaint type and borough
double get_sla_threshold(const char* complaint_type, const char* borough, int is_weekend) {
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
    if (strcmp(borough, "MANHATTAN") == 0 || strcmp(borough, "BROOKLYN") == 0) {
        base_hours *= 0.8;
    }

    // Weekend complaints get extended SLA
    if (is_weekend) {
        base_hours *= 1.5;
    }

    return base_hours;
}

// Count violations in same ZIP within rolling 7-day window (read-only on shared array)
int count_rolling_violations(Complaint* complaints, int total, int current_idx) {
    int count = 0;
    const char* current_zip = complaints[current_idx].incident_zip;
    time_t current_time = complaints[current_idx].created_date;

    for (int i = 0; i < total; i++) {
        if (i == current_idx) continue;

        // Same ZIP
        if (strcmp(complaints[i].incident_zip, current_zip) != 0) continue;

        time_t other_time = complaints[i].created_date;
        double days_diff = difftime(current_time, other_time) / (24.0 * 3600.0);

        if (days_diff >= 0 && days_diff <= ROLLING_WINDOW_DAYS) {
            double sla = get_sla_threshold(complaints[i].complaint_type,
                                           complaints[i].borough,
                                           complaints[i].is_weekend);
            if (complaints[i].response_hours > sla) {
                count++;
            }
        }
    }

    return count;
}

// Load CSV file into complaints[]
int load_csv_file(const char* filename, Complaint* complaints, int* count) {
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
        while (token) {
            switch (col) {
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

// Thread worker: each thread processes [start_idx, end_idx) of complaints[]
void* thread_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    Complaint* complaints = args->complaints;
    int total   = args->total;
    int start_i = args->start_idx;
    int end_i   = args->end_idx;

    int local_viol = 0;
    int local_storms = 0;

    for (int i = start_i; i < end_i; i++) {
        double sla = get_sla_threshold(complaints[i].complaint_type,
                                       complaints[i].borough,
                                       complaints[i].is_weekend);

        if (complaints[i].response_hours > sla) {
            local_viol++;

            int rolling_count = count_rolling_violations(complaints, total, i);
            if (rolling_count >= 5) {
                local_storms++;
            }
        }

        // Optional: local progress (commented to avoid spam)
        // if (((i - start_i + 1) % 100000) == 0) {
        //     printf("Thread [%d-%d): processed %d records...\n",
        //            start_i, end_i, i - start_i + 1);
        // }
    }

    args->local_violations = local_viol;
    args->local_storms     = local_storms;

    return NULL;
}

int main(int argc, char* argv[]) {
    double total_start = now_seconds();

    int num_threads = 4;
    if (argc >= 2) {
        int requested = atoi(argv[1]);
        if (requested > 0) num_threads = requested;
    }

    Complaint* complaints = (Complaint*)malloc(MAX_COMPLAINTS * sizeof(Complaint));
    if (!complaints) {
        printf("Error: Memory allocation failed\n");
        return 1;
    }

    int count = 0;

    printf("=== LOADING DATA (SEQUENTIAL) ===\n");

    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    int gret = glob("data-2025-10.csv", 0, NULL, &glob_result);
    if (gret != 0) {
        printf("Error: glob failed to find data-2025-10.csv\n");
        free(complaints);
        return 1;
    }

    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        printf("Loading %s...\n", glob_result.gl_pathv[i]);
        int loaded = load_csv_file(glob_result.gl_pathv[i], complaints, &count);
        printf("  Loaded %d records (Total: %d)\n", loaded, count);
    }

    globfree(&glob_result);

    if (count == 0) {
        printf("Error: No data loaded\n");
        free(complaints);
        return 1;
    }

    double load_end = now_seconds();
    double load_time = load_end - total_start;

    printf("\nTotal loaded: %d complaints\n", count);
    printf("Load time: %.2f seconds\n", load_time);

    printf("\n=== STARTING PTHREADS COMPUTATION ===\n");
    printf("Using %d threads\n", num_threads);

    double process_start = now_seconds();

    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* targs  = (ThreadArgs*)malloc(num_threads * sizeof(ThreadArgs));
    if (!threads || !targs) {
        printf("Error: Thread allocation failed\n");
        free(complaints);
        free(threads);
        free(targs);
        return 1;
    }

    // Partition the [0, count) range among threads
    for (int t = 0; t < num_threads; t++) {
        int start_idx = (t * count) / num_threads;
        int end_idx   = ((t + 1) * count) / num_threads;

        targs[t].complaints = complaints;
        targs[t].total = count;
        targs[t].start_idx = start_idx;
        targs[t].end_idx   = end_idx;
        targs[t].local_violations = 0;
        targs[t].local_storms     = 0;

        if (pthread_create(&threads[t], NULL, thread_worker, &targs[t]) != 0) {
            printf("Error: Failed to create thread %d\n", t);
            // Simple abort if thread creation fails
            num_threads = t;
            break;
        }
    }

    int total_violations = 0;
    int storm_complaints = 0;

    // Join threads and accumulate their results
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
        total_violations += targs[t].local_violations;
        storm_complaints += targs[t].local_storms;
    }

    double process_end = now_seconds();
    double process_time = process_end - process_start;
    double total_time = process_end - total_start;

    printf("\n=== RESULTS ===\n");
    printf("Total complaints: %d\n", count);
    printf("SLA violations: %d (%.2f%%)\n",
           total_violations,
           (double)total_violations / (double)count * 100.0);
    printf("Complaint storms (5+ violations in 7-day window): %d\n",
           storm_complaints);

    printf("\n=== TIMING ===\n");
    printf("Threads used: %d\n", num_threads);
    printf("Load time: %.2f seconds\n", load_time);
    printf("Computation time: %.2f seconds\n", process_time);
    printf("Total time: %.2f seconds\n", total_time);

    free(complaints);
    free(threads);
    free(targs);

    return 0;
}
