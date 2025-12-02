"""
NYC 311 Service Requests Data Extraction Script
Extracts the last 6 months of 311 data using Socrata API with robust error handling
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from sodapy import Socrata
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nyc_311_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
APP_TOKEN = "ytXvZkcYe1EhSvLVRmwwjRV6D"  # Your app token
DOMAIN = "data.cityofnewyork.us"
DATASET_ID = "erm2-nwe9"  # 311 Service Requests dataset
CHUNK_SIZE = 50000  # Records per request (balanced for reliability)
OUTPUT_DIR = "nyc_311_data"
MONTHS_TO_EXTRACT = 6

class NYC311Extractor:
    def __init__(self, app_token, domain, dataset_id, output_dir):
        """Initialize the extractor with Socrata client and retry logic"""
        self.app_token = app_token
        self.domain = domain
        self.dataset_id = dataset_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Socrata client
        self.client = Socrata(
            domain,
            app_token,
            timeout=300  # 5 minute timeout
        )
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        
        logger.info(f"Initialized NYC 311 Extractor - Output: {self.output_dir}")
    
    def calculate_date_range(self, months=6):
        """Calculate the date range for the last N months"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)  # Approximate
        
        # Format for SoQL (Socrata Query Language)
        start_str = start_date.strftime("%Y-%m-%dT00:00:00.000")
        end_str = end_date.strftime("%Y-%m-%dT23:59:59.999")
        
        logger.info(f"Date range: {start_str} to {end_str}")
        return start_str, end_str
    
    def get_total_count(self, start_date, end_date):
        """Get the total count of records in the date range"""
        where_clause = f"created_date between '{start_date}' and '{end_date}'"
        
        try:
            result = self.client.get(
                self.dataset_id,
                select="COUNT(*) as count",
                where=where_clause
            )
            total = int(result[0]['count'])
            logger.info(f"Total records to extract: {total:,}")
            return total
        except Exception as e:
            logger.error(f"Error getting total count: {e}")
            raise
    
    def extract_chunk(self, offset, limit, start_date, end_date, retry_count=3):
        """Extract a chunk of data with retry logic"""
        where_clause = f"created_date between '{start_date}' and '{end_date}'"
        
        for attempt in range(retry_count):
            try:
                logger.info(f"Fetching offset {offset:,} (limit: {limit:,}) - Attempt {attempt + 1}")
                
                results = self.client.get(
                    self.dataset_id,
                    where=where_clause,
                    order="created_date ASC",  # Stable ordering for pagination
                    limit=limit,
                    offset=offset
                )
                
                if not results:
                    logger.warning(f"No results returned for offset {offset}")
                    return None
                
                df = pd.DataFrame.from_records(results)
                logger.info(f"Successfully fetched {len(df):,} records")
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch offset {offset} after {retry_count} attempts")
                    raise
    
    def save_chunk(self, df, chunk_number):
        """Save a chunk to parquet format (efficient for Dask)"""
        filename = self.output_dir / f"nyc_311_chunk_{chunk_number:04d}.parquet"
        df.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"Saved chunk {chunk_number} to {filename}")
    
    def extract_all(self, months=6):
        """Main extraction logic"""
        start_time = datetime.now()
        logger.info("="*80)
        logger.info("Starting NYC 311 Data Extraction")
        logger.info("="*80)
        
        # Calculate date range
        start_date, end_date = self.calculate_date_range(months)
        
        # Get total count
        try:
            total_records = self.get_total_count(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get total count: {e}")
            return False
        
        # Calculate number of chunks
        num_chunks = (total_records // CHUNK_SIZE) + 1
        logger.info(f"Will extract {num_chunks} chunks of {CHUNK_SIZE:,} records each")
        
        # Extract data in chunks
        successful_chunks = 0
        failed_chunks = []
        
        for chunk_num in range(num_chunks):
            offset = chunk_num * CHUNK_SIZE
            
            try:
                # Extract chunk
                df = self.extract_chunk(offset, CHUNK_SIZE, start_date, end_date)
                
                if df is not None and len(df) > 0:
                    # Save chunk
                    self.save_chunk(df, chunk_num)
                    successful_chunks += 1
                    
                    # Progress update
                    progress = (successful_chunks / num_chunks) * 100
                    logger.info(f"Progress: {progress:.1f}% ({successful_chunks}/{num_chunks} chunks)")
                    
                    # Rate limiting - be nice to the API
                    time.sleep(1)
                else:
                    logger.warning(f"Chunk {chunk_num} was empty, stopping extraction")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_num}: {e}")
                failed_chunks.append(chunk_num)
                continue
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("="*80)
        logger.info("Extraction Complete")
        logger.info("="*80)
        logger.info(f"Total time: {duration}")
        logger.info(f"Successful chunks: {successful_chunks}/{num_chunks}")
        logger.info(f"Failed chunks: {len(failed_chunks)}")
        if failed_chunks:
            logger.warning(f"Failed chunk numbers: {failed_chunks}")
        logger.info(f"Data saved to: {self.output_dir}")
        logger.info("="*80)
        
        return len(failed_chunks) == 0
    
    def verify_extraction(self):
        """Verify the extracted data"""
        logger.info("Verifying extracted data...")
        
        parquet_files = list(self.output_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.error("No parquet files found!")
            return False
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Count total records
        total_records = 0
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                total_records += len(df)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
        
        logger.info(f"Total records across all files: {total_records:,}")
        
        # Sample check
        if parquet_files:
            sample_file = parquet_files[0]
            df = pd.read_parquet(sample_file)
            logger.info(f"\nSample data from {sample_file.name}:")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        
        return True
    
    def close(self):
        """Close the Socrata client"""
        self.client.close()
        logger.info("Closed Socrata client")


def main():
    """Main execution function"""
    try:
        # Initialize extractor
        extractor = NYC311Extractor(
            app_token=APP_TOKEN,
            domain=DOMAIN,
            dataset_id=DATASET_ID,
            output_dir=OUTPUT_DIR
        )
        
        # Extract data
        success = extractor.extract_all(months=MONTHS_TO_EXTRACT)
        
        # Verify extraction
        if success:
            extractor.verify_extraction()
        
        # Cleanup
        extractor.close()
        
        if success:
            logger.info("\n✅ Extraction completed successfully!")
            logger.info(f"📁 Data saved to: {OUTPUT_DIR}")
            logger.info("\n📊 Next steps:")
            logger.info("   1. Use Dask to read the parquet files:")
            logger.info("      import dask.dataframe as dd")
            logger.info(f"      df = dd.read_parquet('{OUTPUT_DIR}/*.parquet')")
            logger.info("   2. Perform your ETL and analysis")
        else:
            logger.error("\n❌ Extraction completed with errors. Check the log file.")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Extraction interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()