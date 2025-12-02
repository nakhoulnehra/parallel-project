#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <glob.h>
#include <mpi.h>

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

// Get SLA threshold based on type/borough/weekend
double get_sla_threshold(const char* complaint_type, const char* borough, int is_weekend) {
    double base_hours = 72.0; // 3 days

    if (strstr(complaint_type, "Food Poisoning") ||
        strstr(complaint_type, "Unsanitary") ||
        strstr(complaint_type, "Rodent")) {
        base_hours = 24.0;
    } else if (strstr(complaint_type, "Noise") ||
               strstr(complaint_type, "Heat/Hot Water")) {
        base_hours = 48.0;
    }

    if (strcmp(borough, "MANHATTAN") == 0 || strcmp(borough, "BROOKLYN") == 0)
        base_hours *= 0.8;

    if (is_weekend)
        base_hours *= 1.5;

    return base_hours;
}

// Count violations in same ZIP in 7-day window
int count_rolling_violations(Complaint* complaints, int total, int current_idx) {
    int count = 0;
    const char* current_zip = complaints[current_idx].incident_zip;
    time_t current_time = complaints[current_idx].created_date;

    for (int i = 0; i < total; i++) {
        if (i == current_idx) continue;

        if (strcmp(complaints[i].incident_zip, current_zip) != 0) continue;

        double days_diff =
            difftime(current_time, complaints[i].created_date) / (24.0 * 3600.0);

        if (days_diff >= 0 && days_diff <= ROLLING_WINDOW_DAYS) {
            double sla = get_sla_threshold(
                complaints[i].complaint_type,
                complaints[i].borough,
                complaints[i].is_weekend
            );
            if (complaints[i].response_hours > sla)
                count++;
        }
    }
    return count;
}

// Load CSV file
int load_csv_file(const char* filename, Complaint* complaints, int* count) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Warning: Cannot open file %s\n", filename);
        return 0;
    }

    char line[MAX_LINE];

    // Skip header (we ignore the return value on purpose)
    fgets(line, MAX_LINE, fp);

    int loaded = 0;
    while (fgets(line, MAX_LINE, fp) && *count < MAX_COMPLAINTS) {
        char* token;
        int col = 0;

        token = strtok(line, ",");
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
            token = strtok(NULL, ",");
            col++;
        }

        (*count)++;
        loaded++;
    }

    fclose(fp);
    return loaded;
}

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double total_start = MPI_Wtime();

    Complaint* complaints = malloc(MAX_COMPLAINTS * sizeof(Complaint));
    if (!complaints) {
        if (rank == 0) printf("Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int count = 0;
    double load_time = 0.0;

    // Rank 0 loads CSV
    if (rank == 0) {
        printf("=== LOADING DATA (Rank 0) ===\n");

        glob_t glob_result;
        memset(&glob_result, 0, sizeof(glob_result));

        int gret = glob("data-2025-10.csv", 0, NULL, &glob_result);
        if (gret != 0) {
            printf("Error: Could not find data-2025-10.csv\n");
            free(complaints);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (size_t i = 0; i < glob_result.gl_pathc; i++) {
            printf("Loading %s...\n", glob_result.gl_pathv[i]);
            int loaded = load_csv_file(glob_result.gl_pathv[i], complaints, &count);
            printf("  Loaded %d (Total: %d)\n", loaded, count);
        }

        globfree(&glob_result);

        load_time = MPI_Wtime() - total_start;

        printf("\nTotal loaded: %d complaints\n", count);
        printf("Load time: %.2f sec\n", load_time);
    }

    // Broadcast count to all ranks
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (count == 0) {
        if (rank == 0) {
            printf("No complaints loaded, exiting.\n");
        }
        free(complaints);
        MPI_Finalize();
        return 0;
    }

    // Broadcast complaints array (as raw bytes)
    MPI_Bcast(complaints, count * (int)sizeof(Complaint), MPI_BYTE, 0, MPI_COMM_WORLD);

    double process_start = MPI_Wtime();

    int start_idx = (rank * count) / size;
    int end_idx   = ((rank + 1) * count) / size;

    int local_viol   = 0;
    int local_storms = 0;

    for (int i = start_idx; i < end_idx; i++) {
        double sla = get_sla_threshold(
            complaints[i].complaint_type,
            complaints[i].borough,
            complaints[i].is_weekend
        );

        if (complaints[i].response_hours > sla) {
            local_viol++;
            int rc = count_rolling_violations(complaints, count, i);
            if (rc >= 5) local_storms++;
        }
    }

    int total_viol   = 0;
    int total_storms = 0;

    MPI_Reduce(&local_viol,   &total_viol,   1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_storms, &total_storms, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    double process_end = MPI_Wtime();
    double process_time_local = process_end - process_start;
    double process_time_max = 0.0;

    MPI_Reduce(&process_time_local, &process_time_max,
               1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== RESULTS ===\n");
        printf("Total complaints: %d\n", count);
        printf("SLA violations: %d (%.2f%%)\n",
               total_viol, 100.0 * total_viol / count);
        printf("Complaint storms (>=5 in 7 days): %d\n", total_storms);

        printf("\n=== TIMING ===\n");
        printf("Load time: %.2f sec\n", load_time);
        printf("Max computation time: %.2f sec\n", process_time_max);
        printf("Total wall time: %.2f sec\n", MPI_Wtime() - total_start);
    }

    free(complaints);
    MPI_Finalize();
    return 0;
}
