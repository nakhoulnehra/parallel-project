// File: sequential.c — NYC311 header-driven CSV clustering with heavy analytics

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

// Section: Configuration constants and thresholds
#define MAX_LINE 8192                   // Maximum size of a single CSV line buffer
#define MAX_RECORDS 2500000             // Maximum number of records to load into memory
#define EARTH_RADIUS_KM 6371.0          // Earth radius in kilometers

#define SPATIAL_THRESHOLD_KM 5.0        // Spatial kernel scale (km) for similarity decay
#define MIN_CLUSTER_SIZE 3              // Minimum members to accept a cluster
static const double SIM_THRESHOLD = 50.0;  // Similarity threshold to link two points

#define SEARCH_SPATIAL_KM   15.0        // Candidate scan longitude window radius (km)
#define LAT_PAD_KM          20.0        // Sliding window half-width in latitude (km)
#define HOUR_PREFILTER      12          // Max absolute hour difference for prefilter
#define DOW_PREFILTER        3          // Max absolute day-of-week difference for prefilter
#define COMPLAINT_PREFIX_N   0          // Complaint-type prefix length for prefilter (0=off)

#define SIL_NEIGHBOR_KM     30          // Silhouette: centroid search radius (km)
#define SIL_USE_HAVERSINE   1           // Silhouette: 1=haversine distance, 0=equirectangular

#define HIST_ENABLE          1          // Pairwise histogram median enabled toggle
#define HIST_TOPK            8          // Number of largest clusters to process for median
#define HIST_MAX_MEMBERS     8000       // Cap of members per processed cluster for O(m^2)
#define HIST_MAX_DIST_KM     10.0       // Histogram distance range (km)
#define HIST_BIN_KM          0.25       // Histogram bin width (km)

// Section: Data types for input records and cluster statistics
typedef struct {
    char unique_key[32];        // Record unique identifier from CSV
    char created_date[32];      // Created date/time string from CSV
    char complaint_type[128];   // Complaint type/category
    char agency[16];            // Reporting/handling agency
    char borough[32];           // Borough name
    double latitude;            // Latitude in degrees
    double longitude;           // Longitude in degrees
    int created_hour;           // Hour of day [0..23]
    int created_dow;            // Day of week [0..6]
    double response_hours;      // Response time in hours
    int cluster_id;             // Assigned cluster id (-1 if unassigned)
    double lat_rad;             // Latitude in radians (precomputed)
    double lon_rad;             // Longitude in radians (precomputed)
    double cos_lat;             // Cosine of latitude (precomputed)
} ServiceRequest;

typedef struct {
    int cluster_id;             // Cluster identifier
    int member_count;           // Number of members in cluster
    double centroid_lat;        // Centroid latitude (degrees)
    double centroid_lon;        // Centroid longitude (degrees)
    double avg_response_hours;  // Mean response hours over members
    int peak_hour;              // Most frequent created hour
    char dominant_complaint[128]; // Majority complaint type label

    double avg_dist_km;         // Mean distance to centroid (km)
    double silhouette;          // Approx silhouette score [-1..1]
    double med_pair_km;         // Median pairwise distance (km) for top clusters
    double axis_ratio;          // Shape anisotropy sqrt(lambda_max/lambda_min)
} ClusterStats;

// Section: Small helpers (math, string trimming/lowercasing, and distances)
static inline double deg_to_rad(double d){ return d * 3.14159265358979323846 / 180.0; } // Convert degrees to radians

static inline void trim_ws(char* s) { // Trim whitespace and surrounding quotes in-place
    size_t n = strlen(s);
    while (n && (s[n-1]==' '||s[n-1]=='\t'||s[n-1]=='\r'||s[n-1]=='\n')) s[--n]='\0';
    size_t i=0; while (s[i]==' '||s[i]=='\t') i++; if (i) memmove(s, s+i, strlen(s+i)+1);
    n = strlen(s);
    if (n>=2 && s[0]=='"' && s[n-1]=='"') { s[n-1]='\0'; memmove(s, s+1, n); }
}

static inline void lower_inplace(char* s){ for (; *s; ++s) *s = (char)tolower((unsigned char)*s); } // Lowercase a string

static inline double dist_km_equirect( // Equirectangular approximation distance (km)
    double lat1r, double lon1r, double coslat1,
    double lat2r, double lon2r, double coslat2)
{
    double cos_avg = 0.5 * (coslat1 + coslat2);
    double x = (lon2r - lon1r) * cos_avg;
    double y = (lat2r - lat1r);
    double d = sqrt(x*x + y*y);
    return EARTH_RADIUS_KM * d;
}

static inline double dist_km_haversine( // Haversine distance (km)
    double lat1r, double lon1r, double lat2r, double lon2r)
{
    double dlat = lat2r - lat1r;
    double dlon = lon2r - lon1r;
    double a = sin(dlat/2.0)*sin(dlat/2.0) +
               cos(lat1r)*cos(lat2r)*sin(dlon/2.0)*sin(dlon/2.0);
    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
    return EARTH_RADIUS_KM * c;
}

static inline double complaint_prefix_similarity(const char* a, const char* b) { // Asymmetric true-prefix similarity
    int L1 = (int)strlen(a), L2 = (int)strlen(b), L = (L1 < L2 ? L1 : L2);
    int match = 0; for (int i=0; i<L; i++){ if (a[i]==b[i]) match++; else break; }
    if (L1 <= 0) return 0.0; return (double)match / (double)L1;
}

static inline void majority_vote_update(char* cur, int* cnt, const char* val) { // Boyer–Moore majority vote update
    if (*cnt == 0) { strncpy(cur, val, 127); cur[127]='\0'; *cnt = 1; }
    else if (strcmp(cur, val) == 0) (*cnt)++; else (*cnt)--;
}

// Section: CSV parsing (header-driven, robust to column order)
typedef struct {
    int unique_key;       // Column index for unique_key
    int created_date;     // Column index for created_date
    int complaint_type;   // Column index for complaint_type
    int agency;           // Column index for agency
    int borough;          // Column index for borough
    int latitude;         // Column index for latitude
    int longitude;        // Column index for longitude
    int created_dow;      // Column index for created_dow
    int created_hour;     // Column index for created_hour
    int response_hours;   // Column index for response_hours
} HeaderMap;

static void init_header_map(HeaderMap* m) { // Initialize header map slots to -1
    m->unique_key = m->created_date = m->complaint_type = m->agency = m->borough = -1;
    m->latitude = m->longitude = m->created_dow = m->created_hour = m->response_hours = -1;
}

static void build_header_map(const char* header_line, HeaderMap* map, char delimiter) { // Parse header row to fill map
    char buf[MAX_LINE]; strncpy(buf, header_line, MAX_LINE-1); buf[MAX_LINE-1]='\0';
    int idx=0; int inq=0; char* p=buf; char* start=p; char save=0; char name[256];
    init_header_map(map);
    for (;; ++p) {
        char c=*p; if (c=='"') inq=!inq;
        if (c=='\0' || (c==delimiter && !inq)) {
            save=*p; *p='\0';
            strncpy(name, start, 255); name[255]='\0';
            trim_ws(name); lower_inplace(name);
            if      (!strcmp(name, "unique_key"))     map->unique_key=idx;
            else if (!strcmp(name, "created_date"))   map->created_date=idx;
            else if (!strcmp(name, "complaint_type")) map->complaint_type=idx;
            else if (!strcmp(name, "agency"))         map->agency=idx;
            else if (!strcmp(name, "borough"))        map->borough=idx;
            else if (!strcmp(name, "latitude"))       map->latitude=idx;
            else if (!strcmp(name, "longitude"))      map->longitude=idx;
            else if (!strcmp(name, "created_dow"))    map->created_dow=idx;
            else if (!strcmp(name, "created_hour"))   map->created_hour=idx;
            else if (!strcmp(name, "response_hours")) map->response_hours=idx;
            if (save=='\0') break;
            *p=save; idx++; start=p+1;
        }
    }
}

static int parse_row_into(const char* line_in, const HeaderMap* map, ServiceRequest* r, char delimiter) { // Parse one CSV row
    char line[MAX_LINE]; strncpy(line, line_in, MAX_LINE-1); line[MAX_LINE-1]='\0';
    memset(r, 0, sizeof(*r));
    r->cluster_id=-1; r->created_hour=r->created_dow=-1;
    int idx=0; int inq=0; char* p=line; char* start=p; char save=0; char tok[MAX_LINE];
    for (;; ++p) {
        char c=*p; if (c=='"') inq=!inq;
        if (c=='\0' || (c==delimiter && !inq)) {
            save=*p; *p='\0';
            strncpy(tok, start, MAX_LINE-1); tok[MAX_LINE-1]='\0';
            trim_ws(tok);
            if (idx==map->unique_key)          { strncpy(r->unique_key, tok, 31); r->unique_key[31]='\0'; }
            else if (idx==map->created_date)   { strncpy(r->created_date, tok, 31); r->created_date[31]='\0'; }
            else if (idx==map->complaint_type) { strncpy(r->complaint_type, tok, 127); r->complaint_type[127]='\0'; }
            else if (idx==map->agency)         { strncpy(r->agency, tok, 15); r->agency[15]='\0'; }
            else if (idx==map->borough)        { strncpy(r->borough, tok, 31); r->borough[31]='\0'; }
            else if (idx==map->latitude)       { if (*tok) r->latitude=atof(tok); }
            else if (idx==map->longitude)      { if (*tok) r->longitude=atof(tok); }
            else if (idx==map->created_dow)    { if (*tok) r->created_dow=atoi(tok); }
            else if (idx==map->created_hour)   { if (*tok) r->created_hour=atoi(tok); }
            else if (idx==map->response_hours) { if (*tok) r->response_hours=atof(tok); }
            if (save=='\0') break; *p=save; idx++; start=p+1;
        }
    }
    r->lat_rad = deg_to_rad(r->latitude);
    r->lon_rad = deg_to_rad(r->longitude);
    r->cos_lat = cos(r->lat_rad);
    return 1;
}

int load_data_one_file(const char* filename, ServiceRequest* requests, int max_records) { // Load one CSV file into array
    FILE* f=fopen(filename,"r"); if(!f){ printf("Error: cannot open %s\n", filename); return 0; }
    char line[MAX_LINE]; if(!fgets(line, MAX_LINE, f)){ fclose(f); return 0; }
    HeaderMap map; build_header_map(line, &map, ',');
    const char* need=NULL; // Name of a missing required column (if any)
    if (map.latitude<0 || map.longitude<0) need="latitude/longitude";
    else if (map.created_hour<0)           need="created_hour";
    else if (map.created_dow<0)            need="created_dow";
    else if (map.response_hours<0)         need="response_hours";
    else if (map.complaint_type<0)         need="complaint_type";
    else if (map.agency<0)                 need="agency";
    else if (map.borough<0)                need="borough";
    if (need){ printf("Error: required column missing: %s\n", need); fclose(f); return 0; }

    int count=0;            // Number of successfully parsed records
    long total_lines=0;     // Number of data lines encountered (excluding header)
    while (fgets(line, MAX_LINE, f) && count<max_records) {
        total_lines++; parse_row_into(line, &map, &requests[count], ','); count++;
    }
    fclose(f);
    printf("Loaded %d records from %s (data rows after header: %ld)\n", count, filename, total_lines);
    return count;
}

// Section: Similarity scoring and prefilters to reduce candidate pairs
static inline int temporal_prefilter(int h1, int h2, int d1, int d2) { // Hour/DOW ring distance prefilter
    int hd=abs(h1-h2); if(hd>12) hd=24-hd; if(hd>HOUR_PREFILTER) return 0;
    int dd=abs(d1-d2); if(dd>3) dd=7-dd;   if(dd>DOW_PREFILTER)  return 0;
    return 1;
}

static inline int complaint_prefilter(const char* a, const char* b) { // Complaint-type prefix prefilter
    if (COMPLAINT_PREFIX_N<=0) return 1;
    for(int i=0;i<COMPLAINT_PREFIX_N;i++){ if(a[i]=='\0'||b[i]=='\0') return 1; if(a[i]!=b[i]) return 0; }
    return 1;
}

static inline double calculate_similarity_fast(const ServiceRequest* r1, const ServiceRequest* r2) { // Composite similarity
    double spatial_dist = dist_km_equirect(r1->lat_rad, r1->lon_rad, r1->cos_lat,
                                           r2->lat_rad, r2->lon_rad, r2->cos_lat);
    double spatial_weight = exp(-spatial_dist / SPATIAL_THRESHOLD_KM);
    double score = spatial_weight * 40.0;

    int hour_diff = abs(r1->created_hour - r2->created_hour);
    if (hour_diff > 12) hour_diff = 24 - hour_diff;
    int dow_diff = abs(r1->created_dow - r2->created_dow);
    if (dow_diff > 3) dow_diff = 7 - dow_diff;
    double temporal_weight = (12.0 - hour_diff) / 12.0 * (7.0 - dow_diff) / 7.0;
    score += temporal_weight * 25.0;

    double complaint_similarity = complaint_prefix_similarity(r1->complaint_type, r2->complaint_type);
    score += complaint_similarity * 20.0;

    if (strcmp(r1->borough, r2->borough) == 0) score += 10.0;
    if (strcmp(r1->agency,  r2->agency)  == 0) score += 5.0;

    return score;
}

static inline double max_lon_deg_at_lat(double lat_rad) { // Convert SEARCH_SPATIAL_KM to lon degrees at given latitude
    double coslat=cos(lat_rad); if(coslat<1e-6) coslat=1e-6;
    return SEARCH_SPATIAL_KM / (111.32 * coslat);
}

static inline double lat_pad_deg() { return LAT_PAD_KM / 111.32; } // Convert LAT_PAD_KM to degrees

// Section: Clustering (lat-sorted blocked scan with filters)
typedef struct { double lat; int idx; } LatIndex; // Sorted-by-lat index: lat (deg), original index
static int cmp_latindex(const void* a,const void* b){
    double da=((const LatIndex*)a)->lat, db=((const LatIndex*)b)->lat;
    return (da<db)?-1:(da>db)?1:0;
}

void perform_clustering_fast(ServiceRequest* req, int n) { // Assign clusters using sliding windows and similarity threshold
    printf("Starting clustered scan with blocking/sliding window...\n");

    LatIndex* order=(LatIndex*)malloc(sizeof(LatIndex)*n);     // Sorted-by-lat working array
    for(int i=0;i<n;i++){ order[i].lat=req[i].latitude; order[i].idx=i; }
    qsort(order,n,sizeof(LatIndex),cmp_latindex);

    int buf_cap=64;                      // Buffer capacity for neighbor indices
    int* buf=(int*)malloc(sizeof(int)*buf_cap); // Buffer for neighbor indices
    int next_cluster_id=0;               // Next cluster identifier to assign
    double latpad=lat_pad_deg();         // Latitude half-width (degrees) for sliding window
    int wstart=0;                        // Start index of current lat window in 'order'

    for (int ii=0; ii<n; ii++) {
        int i = order[ii].idx; if (req[i].cluster_id != -1) continue;

        double lat_i=req[i].latitude, lon_i=req[i].longitude; // Current point lat/lon (deg)
        while (wstart < ii) {
            int k=order[wstart].idx; if ((lat_i - req[k].latitude) > latpad) wstart++; else break;
        }

        int mcount=0;                                   // Number of candidate neighbors accepted
        double lon_pad=max_lon_deg_at_lat(req[i].lat_rad); // Longitude half-width (deg) at current lat

        for (int jj=ii-1; jj>=wstart; jj--) {           // Backward scan within window
            int j=order[jj].idx; if (req[j].cluster_id!=-1) continue;
            if (fabs(lon_i - req[j].longitude) > lon_pad) continue;
            if (!temporal_prefilter(req[i].created_hour, req[j].created_hour, req[i].created_dow, req[j].created_dow)) continue;
            if (!complaint_prefilter(req[i].complaint_type, req[j].complaint_type)) continue;

            double s = calculate_similarity_fast(&req[i], &req[j]);
            if (s >= SIM_THRESHOLD) { if (mcount>=buf_cap){buf_cap*=2; buf=(int*)realloc(buf,sizeof(int)*buf_cap);} buf[mcount++]=j; }
        }

        for (int jj=ii+1; jj<n; jj++) {                 // Forward scan within window
            int j=order[jj].idx; if ((req[j].latitude - lat_i) > latpad) break;
            if (req[j].cluster_id!=-1) continue;
            if (fabs(lon_i - req[j].longitude) > lon_pad) continue;
            if (!temporal_prefilter(req[i].created_hour, req[j].created_hour, req[i].created_dow, req[j].created_dow)) continue;
            if (!complaint_prefilter(req[i].complaint_type, req[j].complaint_type)) continue;

            double s = calculate_similarity_fast(&req[i], &req[j]);
            if (s >= SIM_THRESHOLD) { if (mcount>=buf_cap){buf_cap*=2; buf=(int*)realloc(buf,sizeof(int)*buf_cap);} buf[mcount++]=j; }
        }

        if (mcount + 1 >= MIN_CLUSTER_SIZE) {           // Commit cluster if size threshold reached
            int cid=next_cluster_id++; req[i].cluster_id=cid;
            for (int t=0;t<mcount;t++){ int j=buf[t]; if (req[j].cluster_id==-1) req[j].cluster_id=cid; }
        }

        if ((ii % 10000) == 0) printf("  processed %d/%d, clusters=%d\n", ii, n, next_cluster_id);
    }
    free(buf); free(order);
    printf("Clustering done.\n");
}

// Section: Basic cluster statistics (centroids, means, peaks, dominant label)
void calculate_cluster_stats(ServiceRequest* req, int n, ClusterStats** out_stats, int* out_num) { // Compute per-cluster stats
    int max_id=-1; for(int i=0;i<n;i++) if(req[i].cluster_id>max_id) max_id=req[i].cluster_id;
    *out_num = max_id+1; if (*out_num<=0){ *out_stats=NULL; return; }

    ClusterStats* stats=(ClusterStats*)calloc(*out_num,sizeof(ClusterStats)); // Output stats array
    int* hour_counts=(int*)calloc((*out_num)*24,sizeof(int));                 // Per-cluster hour histogram
    int* mj_cnt=(int*)calloc(*out_num,sizeof(int));                           // Majority-vote counters
    for(int i=0;i<*out_num;i++){ stats[i].cluster_id=i; stats[i].dominant_complaint[0]='\0'; }

    for(int i=0;i<n;i++){
        int cid=req[i].cluster_id; if(cid<0) continue;
        stats[cid].member_count++;
        stats[cid].centroid_lat       += req[i].latitude;
        stats[cid].centroid_lon       += req[i].longitude;
        stats[cid].avg_response_hours += req[i].response_hours;
        if (req[i].created_hour>=0 && req[i].created_hour<24) hour_counts[cid*24 + req[i].created_hour]++;
        majority_vote_update(stats[cid].dominant_complaint, &mj_cnt[cid], req[i].complaint_type);
    }
    for(int i=0;i<*out_num;i++){
        if (stats[i].member_count>0){
            stats[i].centroid_lat       /= stats[i].member_count;
            stats[i].centroid_lon       /= stats[i].member_count;
            stats[i].avg_response_hours /= stats[i].member_count;
            int best=0,besth=0;
            for(int h=0;h<24;h++){ int v=hour_counts[i*24+h]; if(v>best){best=v; besth=h;} }
            stats[i].peak_hour=besth;
        }
    }
    free(hour_counts); free(mj_cnt);
    *out_stats=stats;
}

// Section: Membership index for clusters (contiguous member ranges)
typedef struct { int* members; int* offsets; } MembershipIndex; // members: row indices; offsets[c]..offsets[c+1]-1 range for cluster c

static MembershipIndex build_membership_index(ServiceRequest* req, int n, int k) { // Build cluster -> member index
    MembershipIndex mi={0};
    if(k<=0) return mi;
    int* counts=(int*)calloc(k,sizeof(int));            // Per-cluster counts
    for(int i=0;i<n;i++){ int c=req[i].cluster_id; if(c>=0) counts[c]++; }
    mi.offsets=(int*)malloc(sizeof(int)*(k+1)); mi.offsets[0]=0;
    for(int c=0;c<k;c++) mi.offsets[c+1]=mi.offsets[c]+counts[c];
    int total=mi.offsets[k];
    mi.members=(int*)malloc(sizeof(int)*total);
    int* cursor=(int*)malloc(sizeof(int)*k); memcpy(cursor,mi.offsets,sizeof(int)*k);
    for(int i=0;i<n;i++){ int c=req[i].cluster_id; if(c<0) continue; mi.members[cursor[c]++]=i; }
    free(cursor); free(counts); return mi;
}

// Section: Helpers for analytics (precompute centroid radians/cos)
typedef struct { double lat_r; double lon_r; double cos_lat; } CenterRad; // Centroid in radians with cos(lat)
static CenterRad* centers_rad_from_stats(ClusterStats* s, int k){ // Convert centroid degrees to radians
    CenterRad* cr=(CenterRad*)malloc(sizeof(CenterRad)*k);
    for(int i=0;i<k;i++){ cr[i].lat_r=deg_to_rad(s[i].centroid_lat); cr[i].lon_r=deg_to_rad(s[i].centroid_lon); cr[i].cos_lat=cos(cr[i].lat_r); }
    return cr;
}

// Section: Axis ratio (anisotropy) via covariance
static void compute_axis_ratio(ServiceRequest* req, ClusterStats* stats, int k, const MembershipIndex* mi){ // Compute axis ratios
    for(int c=0;c<k;c++){
        int off=mi->offsets[c], len=mi->offsets[c+1]-off;
        if (len<2){ stats[c].axis_ratio=1.0; continue; }
        double mx=0,my=0;                       // Mean longitude/latitude
        for(int t=0;t<len;t++){ int i=mi->members[off+t]; mx += req[i].longitude; my += req[i].latitude; }
        mx/=len; my/=len;
        double sxx=0, syy=0, sxy=0;             // Covariance components
        for(int t=0;t<len;t++){ int i=mi->members[off+t]; double x=req[i].longitude-mx, y=req[i].latitude-my; sxx+=x*x; syy+=y*y; sxy+=x*y; }
        sxx/=len; syy/=len; sxy/=len;
        double tr=sxx+syy; double det=sxx*syy - sxy*sxy; // Trace/determinant of covariance
        double disc=tr*tr - 4*det; if(disc<0) disc=0;
        double l1=(tr + sqrt(disc))/2.0, l2=(tr - sqrt(disc))/2.0; // Eigenvalues
        if (l2 <= 1e-12) stats[c].axis_ratio = (l1<=1e-12)?1.0 : 1e6;
        else stats[c].axis_ratio = sqrt(l1/l2);
    }
}

// Section: Compactness and approximate silhouette using centroid windows
static void compute_compactness_and_silhouette(ServiceRequest* req, int n, ClusterStats* stats, int k) { // Compute avg_dist and silhouette
    CenterRad* centers=centers_rad_from_stats(stats,k);

    typedef struct{ double lat; int idx; } CIdx; // Centroid index entry: latitude, cluster id
    CIdx* cidx=(CIdx*)malloc(sizeof(CIdx)*k);
    for(int c=0;c<k;c++){ cidx[c].lat=stats[c].centroid_lat; cidx[c].idx=c; }
    int cmpc(const void*a,const void*b){ double da=((const CIdx*)a)->lat, db=((const CIdx*)b)->lat; return (da<db)?-1:(da>db)?1:0; }
    qsort(cidx,k,sizeof(CIdx),cmpc);

    double sil_lat_pad = SIL_NEIGHBOR_KM / 111.32; // Latitude window (deg)

    double* sum_dist=(double*)calloc(k,sizeof(double)); // Sum of distances to own centroid
    double* sum_silh=(double*)calloc(k,sizeof(double)); // Sum of silhouette values
    int*    cnt=(int*)calloc(k,sizeof(int));            // Counts per cluster

    for(int i=0;i<n;i++){
        int c=req[i].cluster_id; if(c<0) continue;

#if SIL_USE_HAVERSINE
        double a = dist_km_haversine(req[i].lat_rad, req[i].lon_rad, centers[c].lat_r, centers[c].lon_r); // Own-centroid distance
#else
        double a = dist_km_equirect(req[i].lat_rad, req[i].lon_rad, req[i].cos_lat, centers[c].lat_r, centers[c].lon_r, centers[c].cos_lat);
#endif
        double b = 1e12;                                                  // Nearest other-centroid distance
        double plat = req[i].latitude;                                    // Point latitude (deg)
        int lo=0, hi=k-1, mid=0;                                          // Binary search state
        while (lo<hi){ mid=(lo+hi)/2; if (cidx[mid].lat < plat) lo=mid+1; else hi=mid; }
        int center_pos = lo;                                              // Start index near plat

        for(int p=center_pos; p>=0; --p){                                 // Scan backward window
            int cj=cidx[p].idx; double dlat=fabs(cidx[p].lat - plat);
            if (dlat > sil_lat_pad) break;
            if (cj==c) continue;
            double lon_pad = SEARCH_SPATIAL_KM / (111.32 * fmax(1e-6, cos(deg_to_rad(plat))));
            double clon = stats[cj].centroid_lon;
            if (fabs(req[i].longitude - clon) > lon_pad*2) continue;
#if SIL_USE_HAVERSINE
            double d = dist_km_haversine(req[i].lat_rad, req[i].lon_rad, centers[cj].lat_r, centers[cj].lon_r);
#else
            double d = dist_km_equirect(req[i].lat_rad, req[i].lon_rad, req[i].cos_lat, centers[cj].lat_r, centers[cj].lon_r, centers[cj].cos_lat);
#endif
            if (d<b) b=d;
        }
        for(int p=center_pos+1; p<k; ++p){                                 // Scan forward window
            double dlat=fabs(cidx[p].lat - plat);
            if (dlat > sil_lat_pad) break;
            int cj=cidx[p].idx; if (cj==c) continue;
            double lon_pad = SEARCH_SPATIAL_KM / (111.32 * fmax(1e-6, cos(deg_to_rad(plat))));
            double clon = stats[cj].centroid_lon;
            if (fabs(req[i].longitude - clon) > lon_pad*2) continue;
#if SIL_USE_HAVERSINE
            double d = dist_km_haversine(req[i].lat_rad, req[i].lon_rad, centers[cj].lat_r, centers[cj].lon_r);
#else
            double d = dist_km_equirect(req[i].lat_rad, req[i].lon_rad, req[i].cos_lat, centers[cj].lat_r, centers[cj].lon_r, centers[cj].cos_lat);
#endif
            if (d<b) b=d;
        }

        if (b > 1e11 && k>1){                                             // Fallback: full scan
            for(int cj=0;cj<k;cj++){ if(cj==c) continue;
#if SIL_USE_HAVERSINE
                double d = dist_km_haversine(req[i].lat_rad, req[i].lon_rad, centers[cj].lat_r, centers[cj].lon_r);
#else
                double d = dist_km_equirect(req[i].lat_rad, req[i].lon_rad, req[i].cos_lat, centers[cj].lat_r, centers[cj].lon_r, centers[cj].cos_lat);
#endif
                if (d<b) b=d;
            }
        }

        double denom = (a>b? a : b);                                      // Silhouette denominator
        double s = (denom > 1e-12) ? (b - a) / denom : 0.0;               // Silhouette value

        sum_dist[c] += a; sum_silh[c] += s; cnt[c] += 1;
    }

    for(int c=0;c<k;c++){
        if (cnt[c]>0){ stats[c].avg_dist_km = sum_dist[c]/cnt[c]; stats[c].silhouette = sum_silh[c]/cnt[c]; }
        else { stats[c].avg_dist_km=0.0; stats[c].silhouette=0.0; }
    }

    free(centers); free(cidx); free(sum_dist); free(sum_silh); free(cnt);
}

// Section: Pairwise histogram median for top clusters (O(m^2))
static void compute_pairwise_median_for_top_clusters(ServiceRequest* req, ClusterStats* stats, int k, const MembershipIndex* mi) { // Compute med_pair_km
#if HIST_ENABLE
    typedef struct{ int idx; int members; } View; // Cluster view: id and size
    View* v=(View*)malloc(sizeof(View)*k);
    for(int i=0;i<k;i++){ v[i].idx=i; v[i].members=stats[i].member_count; }
    int cmpv(const void*a,const void*b){ int aa=((const View*)a)->members, bb=((const View*)b)->members; return (aa>bb)?-1:(aa<bb)?1:0; }
    qsort(v,k,sizeof(View),cmpv);

    const int NBINS = (int)(HIST_MAX_DIST_KM / HIST_BIN_KM) + 1;   // Number of histogram bins
    unsigned long long* hist=(unsigned long long*)malloc(sizeof(unsigned long long)*NBINS); // Histogram buffer

    int top = (HIST_TOPK<k)?HIST_TOPK:k;                           // Number of clusters to process
    for(int t=0;t<top;t++){
        int cid=v[t].idx;                                          // Current cluster id
        int off=mi->offsets[cid], len=mi->offsets[cid+1]-off;      // Member range
        if (len<2){ stats[cid].med_pair_km=0.0; continue; }
        int m=len; if(m>HIST_MAX_MEMBERS) m=HIST_MAX_MEMBERS;      // Truncated member count

        memset(hist,0,sizeof(unsigned long long)*NBINS);
        for(int i=0;i<m;i++){
            int ii=mi->members[off+i];
            for(int j=i+1;j<m;j++){
                int jj=mi->members[off+j];
                double d=dist_km_equirect(req[ii].lat_rad, req[ii].lon_rad, req[ii].cos_lat,
                                          req[jj].lat_rad, req[jj].lon_rad, req[jj].cos_lat);
                int bin=(int)(d/HIST_BIN_KM);
                if(bin<0) bin=0; if(bin>=NBINS) bin=NBINS-1; hist[bin]++;
            }
        }
        unsigned long long total_pairs=(unsigned long long)m*(unsigned long long)(m-1)/2ULL; // Total number of pairs
        unsigned long long half=(total_pairs+1ULL)/2ULL, cum=0; double med=0.0;              // Median position
        for(int b=0;b<NBINS;b++){ cum+=hist[b]; if(cum>=half){ med=(b+0.5)*HIST_BIN_KM; break; } }
        stats[cid].med_pair_km=med;
    }
    free(hist); free(v);
#else
    (void)req; (void)stats; (void)k; (void)mi;
#endif
}

// Section: Program entry point (load, cluster, analytics, report)
typedef struct { int idx; int members; } ClusterView; // Pair of cluster index and size for sorting
static int cmp_clusterview_desc(const void* a,const void* b){
    int ma=((const ClusterView*)a)->members, mb=((const ClusterView*)b)->members;
    return (ma>mb)?-1:(ma<mb)?1:0;
}

int main(int argc, char** argv) { // Run full pipeline and print summary report
    printf("=== NYC311 Spatiotemporal Clustering (header-driven, blocked + heavy analytics) ===\n");

    ServiceRequest* requests=(ServiceRequest*)malloc(MAX_RECORDS*sizeof(ServiceRequest)); // Main record buffer
    if(!requests){ printf("Memory allocation failed\n"); return 1; }

    struct timespec t0p,t1p,t0a,t1a,t0q,t1q,t0all,t1all; // Timing checkpoints
    clock_gettime(CLOCK_MONOTONIC,&t0all);
    clock_gettime(CLOCK_MONOTONIC,&t0p);

    int total_records=0; // Total number of loaded records
    if (argc>1){
        for(int a=1;a<argc;a++){
            int added=load_data_one_file(argv[a], requests+total_records, MAX_RECORDS-total_records); // Records loaded from arg file
            total_records+=added; if(total_records>=MAX_RECORDS) break;
        }
    } else {
        const char* filename="data-2025-11.csv"; // Default filename if no args
        total_records=load_data_one_file(filename, requests, MAX_RECORDS);
    }

    clock_gettime(CLOCK_MONOTONIC,&t1p);
    double parse_time=(t1p.tv_sec-t0p.tv_sec)+(t1p.tv_nsec-t0p.tv_nsec)/1e9; // Parse phase duration (not printed by request)

    if(total_records<=0){ printf("No records loaded. Exiting.\n"); free(requests); return 1; }

    clock_gettime(CLOCK_MONOTONIC,&t0a);
    perform_clustering_fast(requests, total_records);

    ClusterStats* stats=NULL; int num_clusters=0; // Stats array and number of clusters
    calculate_cluster_stats(requests, total_records, &stats, &num_clusters);

    MembershipIndex mi=build_membership_index(requests, total_records, num_clusters); // Cluster -> members index

    clock_gettime(CLOCK_MONOTONIC,&t0q);
    compute_axis_ratio(requests, stats, num_clusters, &mi);
    compute_compactness_and_silhouette(requests, total_records, stats, num_clusters);
    compute_pairwise_median_for_top_clusters(requests, stats, num_clusters, &mi);
    clock_gettime(CLOCK_MONOTONIC,&t1q);

    free(mi.members); free(mi.offsets);

    clock_gettime(CLOCK_MONOTONIC,&t1a);
    double cluster_time=(t1a.tv_sec-t0a.tv_sec)+(t1a.tv_nsec-t0a.tv_nsec)/1e9; // Clustering+stats duration
    double quality_time=(t1q.tv_sec-t0q.tv_sec)+(t1q.tv_nsec-t0q.tv_nsec)/1e9; // Heavy analytics duration

    clock_gettime(CLOCK_MONOTONIC,&t1all);
    double total_wall=(t1all.tv_sec-t0all.tv_sec)+(t1all.tv_nsec-t0all.tv_nsec)/1e9; // Total wall clock

    ClusterView* view=NULL; int vcount=0; // View for sorting clusters by size
    if(stats && num_clusters>0){
        view=(ClusterView*)malloc(sizeof(ClusterView)*num_clusters);
        for(int i=0;i<num_clusters;i++){ view[vcount].idx=i; view[vcount].members=stats[i].member_count; vcount++; }
        qsort(view,vcount,sizeof(ClusterView),cmp_clusterview_desc);
    }

    // Report: print run stats (without parse-time line per request) and top clusters
    printf("\n=== RESULTS ===\n");
    printf("Records processed: %d\n", total_records);
    printf("Clusters found (size >= %d): %d\n", MIN_CLUSTER_SIZE, num_clusters);
    printf("Clustering + basic stats: %.2f s\n", cluster_time);
    printf("Heavy analytics time: %.2f s\n", quality_time);
    printf("Total wall time: %.2f s (%.2f min)\n", total_wall, total_wall/60.0);

    printf("\nTop 10 Largest Clusters:\n");
    printf("%-8s %-8s %-10s %-10s %-9s %-7s %-10s %-11s %-10s %-10s %-30s\n",
           "Cluster","Members","Latitude","Longitude","AvgHours","PeakHr","AvgDistKm","Silhouette","MedPairKm","AxisRatio","Complaint Type");
    printf("-------------------------------------------------------------------------------------------------------------------------------------\n");
    int printed=0; // Number printed so far
    for(int k=0; k<vcount && printed<10; k++){
        int i=view[k].idx; if(stats[i].member_count<=0) continue;
        printf("%-8d %-8d %-10.6f %-10.6f %-9.2f %-7d %-10.3f %-11.3f %-10.3f %-10.3f %-30s\n",
               stats[i].cluster_id, stats[i].member_count,
               stats[i].centroid_lat, stats[i].centroid_lon,
               stats[i].avg_response_hours, stats[i].peak_hour,
               stats[i].avg_dist_km, stats[i].silhouette, stats[i].med_pair_km, stats[i].axis_ratio,
               stats[i].dominant_complaint);
        printed++;
    }

    free(view); free(stats); free(requests);
    printf("\nDone.\n");
    return 0;
}
