#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <glob.h>
#include <omp.h>

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
    tm.tm_mon -= 1;
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

// Count violations in same ZIP within rolling window
int count_rolling_violations(Complaint* complaints, int total, int current_idx, time_t window_start) {
    int count = 0;
    const char* current_zip = complaints[current_idx].incident_zip;
    time_t current_time = complaints[current_idx].created_date;
    
    for (int i = 0; i < total; i++) {
        if (i == current_idx) continue;
        
        // Check if in same ZIP
        if (strcmp(complaints[i].incident_zip, current_zip) != 0) continue;
        
        // Check if within rolling window (7 days before current complaint)
        time_t other_time = complaints[i].created_date;
        double days_diff = difftime(current_time, other_time) / (24.0 * 3600.0);
        
        if (days_diff >= 0 && days_diff <= ROLLING_WINDOW_DAYS) {
            // Check if it's a violation
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
        while (token && col < 20) {
            switch(col) {
                case 0: strncpy(complaints[*count].unique_key, token, 31); break;
                case 1: complaints[*count].created_date = parse_date(token); break;
                case 2: complaints[*count].closed_date = parse_date(token); break;
                case 3: complaints[*count].response_hours = atof(token); break;
                case 4: strncpy(complaints[*count].status, token, 15); break;
                case 7: strncpy(complaints[*count].complaint_type, token, 127); break;
                case 9: strncpy(complaints[*count].borough, token, 31); break;
                case 11: strncpy(complaints[*count].incident_zip, token, 15); break;
                case 19: complaints[*count].is_weekend = atoi(token); break;
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

int main(int argc, char* argv[]) {
    double total_start = omp_get_wtime();
    
    Complaint* complaints = malloc(MAX_COMPLAINTS * sizeof(Complaint));
    if (!complaints) {
        printf("Error: Memory allocation failed\n");
        return 1;
    }
    
    int count = 0;
    
    printf("=== LOADING DATA ===\n");
    
    // Load all data-2025-*.csv files
    glob_t glob_result;
    glob("data-2025-10.csv", GLOB_TILDE, NULL, &glob_result);
    
    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        printf("Loading %s...\n", glob_result.gl_pathv[i]);
        int loaded = load_csv_file(glob_result.gl_pathv[i], complaints, &count);
        printf("  Loaded %d records (Total: %d)\n", loaded, count);
    }
    
    globfree(&glob_result);
    
    double load_end = omp_get_wtime();
    double load_time = load_end - total_start;
    
    printf("\nTotal loaded: %d complaints\n", count);
    printf("Load time: %.2f seconds\n", load_time);
    
    if (count == 0) {
        printf("Error: No data loaded\n");
        free(complaints);
        return 1;
    }
    
    // Get number of threads
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }
    
    printf("\n=== STARTING PARALLEL COMPUTATION ===\n");
    printf("Using %d OpenMP threads\n", num_threads);
    
    double process_start = omp_get_wtime();
    
    int total_violations = 0;
    int storm_complaints = 0;
    
    // Parallel processing with OpenMP
    #pragma omp parallel for reduction(+:total_violations,storm_complaints) schedule(dynamic, 1000)
    for (int i = 0; i < count; i++) {
        double sla = get_sla_threshold(complaints[i].complaint_type, 
                                       complaints[i].borough, 
                                       complaints[i].is_weekend);
        
        if (complaints[i].response_hours > sla) {
            total_violations++;
            
            // Count violations in same ZIP within rolling window
            int rolling_count = count_rolling_violations(complaints, count, i, 
                                                        complaints[i].created_date - (ROLLING_WINDOW_DAYS * 24 * 3600));
            
            if (rolling_count >= 5) {
                storm_complaints++;
            }
        }
        
        // Progress reporting (only from thread 0 to avoid race conditions)
        if ((i + 1) % 100000 == 0) {
            #pragma omp critical
            {
                printf("Processed %d records...\n", i + 1);
            }
        }
    }
    
    double process_end = omp_get_wtime();
    double process_time = process_end - process_start;
    double total_time = process_end - total_start;
    
    printf("\n=== RESULTS ===\n");
    printf("Total complaints: %d\n", count);
    printf("SLA violations: %d (%.2f%%)\n", total_violations, (double)total_violations/count * 100);
    printf("Complaint storms (5+ violations in 7-day window): %d\n", storm_complaints);
    printf("\n=== TIMING ===\n");
    printf("Threads used: %d\n", num_threads);
    printf("Load time: %.2f seconds\n", load_time);
    printf("Computation time: %.2f seconds\n", process_time);
    printf("Total time: %.2f seconds\n", total_time);
    
    free(complaints);
    
    return 0;
}