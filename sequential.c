#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <glob.h>

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

// Parse ISO 8601 date format
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

// SLA calculation
double get_sla_threshold(const char* complaint_type, const char* borough, int is_weekend) {
    double base_hours = 72.0;

    if (strstr(complaint_type, "Food Poisoning") ||
        strstr(complaint_type, "Unsanitary") ||
        strstr(complaint_type, "Rodent")) {
        base_hours = 24.0;
    } else if (strstr(complaint_type, "Noise") ||
               strstr(complaint_type, "Heat/Hot Water")) {
        base_hours = 48.0;
    }

    if (strcmp(borough, "MANHATTAN") == 0 ||
        strcmp(borough, "BROOKLYN") == 0) {
        base_hours *= 0.8;
    }

    if (is_weekend) {
        base_hours *= 1.5;
    }

    return base_hours;
}

// Count prior violations in same ZIP in 7-day window
int count_rolling_violations(Complaint* complaints, int total, int current_idx) {
    int count = 0;
    const char* current_zip = complaints[current_idx].incident_zip;
    time_t current_time = complaints[current_idx].created_date;

    for (int i = 0; i < total; i++) {
        if (i == current_idx) continue;

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

// Load CSV
int load_csv_file(const char* filename, Complaint* complaints, int* count) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Warning: Cannot open file %s\n", filename);
        return 0;
    }

    char line[MAX_LINE];

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
    double start_time = (double)clock() / CLOCKS_PER_SEC;

    Complaint* complaints = malloc(MAX_COMPLAINTS * sizeof(Complaint));
    if (!complaints) {
        printf("Memory allocation failed\n");
        return 1;
    }

    int count = 0;

    printf("=== LOADING DATA ===\n");

    glob_t glob_result;
    glob("data-2025-10.csv", GLOB_TILDE, NULL, &glob_result);

    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        printf("Loading %s...\n", glob_result.gl_pathv[i]);
        int loaded = load_csv_file(glob_result.gl_pathv[i], complaints, &count);
        printf("Loaded %d (Total: %d)\n", loaded, count);
    }

    globfree(&glob_result);

    printf("\n=== PROCESSING SEQUENTIALLY ===\n");

    int total_violations = 0;
    int storm_complaints = 0;

    for (int i = 0; i < count; i++) {
        double sla = get_sla_threshold(complaints[i].complaint_type,
                                       complaints[i].borough,
                                       complaints[i].is_weekend);

        if (complaints[i].response_hours > sla) {
            total_violations++;

            int rolling = count_rolling_violations(complaints, count, i);

            if (rolling >= 5) {
                storm_complaints++;
            }
        }

        if ((i + 1) % 100000 == 0) {
            printf("Processed %d records...\n", i + 1);
        }
    }

    double end_time = (double)clock() / CLOCKS_PER_SEC;

    printf("\n=== RESULTS ===\n");
    printf("Total complaints: %d\n", count);
    printf("SLA violations: %d\n", total_violations);
    printf("Complaint storms (>=5 in window): %d\n", storm_complaints);
    printf("Total runtime: %.2f seconds\n", (end_time - start_time));

    free(complaints);
    return 0;
}
