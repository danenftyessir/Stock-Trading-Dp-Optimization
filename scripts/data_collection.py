#!/usr/bin/env python3
"""
Data Collection Script for Stock Trading Optimization.
Downloads and processes historical stock data from Alpha Vantage API.

Usage:
    python scripts/data_collection.py --symbols AAPL GOOGL MSFT
    python scripts/data_collection.py --all
    python scripts/data_collection.py --update
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_downloader import StockDataDownloader
from utils.logger import setup_global_logging
from utils.helpers import load_config

def setup_logging():
    """Setup logging for the script."""
    log_config = {
        'log_level': 'INFO',
        'enable_console': True,
        'enable_file': True,
        'log_dir': 'logs',
        'log_file': f'data_collection_{datetime.now().strftime("%Y%m%d")}.log'
    }
    return setup_global_logging(log_config)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download and process stock data for trading optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --symbols AAPL GOOGL MSFT     # Download specific symbols
  %(prog)s --all                         # Download all symbols from config
  %(prog)s --update                      # Update existing data
  %(prog)s --symbols AAPL --force        # Force re-download
  %(prog)s --status                      # Check data status
        """
    )
    
    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group(required=False)
    symbol_group.add_argument(
        '--symbols', 
        nargs='+', 
        help='Stock symbols to download (e.g., AAPL GOOGL MSFT)'
    )
    symbol_group.add_argument(
        '--all', 
        action='store_true',
        help='Download all symbols specified in config file'
    )
    
    # Action options
    parser.add_argument(
        '--update', 
        action='store_true',
        help='Update existing data with recent data'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force re-download even if data exists'
    )
    parser.add_argument(
        '--status', 
        action='store_true',
        help='Show status of downloaded data'
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    parser.add_argument(
        '--max-stocks', 
        type=int,
        help='Maximum number of stocks to download (useful for testing)'
    )
    
    # Output options
    parser.add_argument(
        '--export', 
        choices=['csv', 'excel', 'json'],
        help='Export processed data in specified format'
    )
    parser.add_argument(
        '--no-technical', 
        action='store_true',
        help='Skip technical indicators calculation'
    )
    
    # Utility options
    parser.add_argument(
        '--cleanup', 
        type=int, 
        metavar='DAYS',
        help='Clean up cache files older than DAYS'
    )
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_config(config_path: str) -> bool:
    """Validate that configuration file exists and contains required keys."""
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("Please create config.yaml or specify correct path with --config")
        return False
    
    try:
        config = load_config(config_path)
        
        # Check required sections
        required_keys = ['api.alpha_vantage.api_key', 'data.tickers']
        for key in required_keys:
            keys = key.split('.')
            current = config
            for k in keys:
                if k not in current:
                    print(f"❌ Missing required configuration key: {key}")
                    return False
                current = current[k]
        
        # Validate API key
        api_key = config['api']['alpha_vantage']['api_key']
        if not api_key or api_key.startswith('YOUR_'):
            print("❌ Please set your Alpha Vantage API key in config.yaml")
            print("Get your free API key from: https://www.alphavantage.co/support/#api-key")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False


def print_data_status(downloader: StockDataDownloader, symbols: list):
    """Print formatted data status table."""
    status = downloader.get_data_status(symbols)
    
    print("\n📊 Data Status Summary")
    print("=" * 80)
    print(f"{'Symbol':<8} {'Status':<12} {'Last Update':<12} {'Total Days':<12} {'Size':<8}")
    print("-" * 80)
    
    for symbol, info in status.items():
        status_str = "✅ Ready" if info['processed_data_exists'] else "❌ Missing"
        last_update = info.get('last_update', 'N/A')[:10]  # Truncate to date only
        total_days = info.get('total_days', 0)
        
        # Calculate file size
        size_str = "N/A"
        if 'file_size_mb' in info:
            size_mb = info['file_size_mb']
            if size_mb < 1:
                size_str = f"{size_mb*1024:.0f}KB"
            else:
                size_str = f"{size_mb:.1f}MB"
        
        print(f"{symbol:<8} {status_str:<12} {last_update:<12} {total_days:<12} {size_str:<8}")
    
    # Summary statistics
    total_symbols = len(status)
    ready_symbols = sum(1 for info in status.values() if info['processed_data_exists'])
    
    print("-" * 80)
    print(f"Summary: {ready_symbols}/{total_symbols} symbols ready "
          f"({ready_symbols/total_symbols*100:.1f}%)")
    print()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate configuration first
    if not validate_config(args.config):
        return 1
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("STOCK DATA COLLECTION SCRIPT STARTED")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Initialize downloader
        downloader = StockDataDownloader(args.config)
        logger.info("Data downloader initialized")
        
        # Determine symbols to process
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
            logger.info(f"Processing specified symbols: {symbols}")
        elif args.all or not (args.status or args.cleanup is not None):
            symbols = config['data']['tickers']
            logger.info(f"Processing all symbols from config: {symbols}")
        else:
            symbols = config['data']['tickers']
        
        # Apply max stocks limit
        if args.max_stocks and len(symbols) > args.max_stocks:
            original_count = len(symbols)
            symbols = symbols[:args.max_stocks]
            logger.info(f"Limited to {args.max_stocks} stocks (from {original_count})")
        
        # Execute requested action
        if args.status:
            print_data_status(downloader, symbols)
            
        elif args.cleanup is not None:
            logger.info(f"Cleaning up cache files older than {args.cleanup} days")
            downloader.cleanup_old_cache(args.cleanup)
            
        elif args.update:
            logger.info("Updating existing data with recent data")
            for symbol in symbols:
                logger.info(f"Updating {symbol}...")
                try:
                    updated_data = downloader.update_existing_data(symbol)
                    if updated_data is not None:
                        logger.info(f"✅ {symbol} updated successfully")
                    else:
                        logger.warning(f"⚠️ {symbol} update failed")
                except Exception as e:
                    logger.error(f"❌ {symbol} update error: {e}")
                    
        elif args.export:
            logger.info(f"Exporting data in {args.export} format")
            export_path = downloader.export_data_for_analysis(
                symbols=symbols,
                format=args.export,
                include_technical_indicators=not args.no_technical
            )
            logger.info(f"✅ Data exported to: {export_path}")
            
        else:
            # Default: download data
            logger.info("Starting data download and processing")
            
            results = downloader.download_multiple_stocks(
                symbols=symbols,
                force_download=args.force,
                max_stocks_per_day=args.max_stocks
            )
            
            # Print results summary
            successful = len(results)
            total = len(symbols)
            
            print(f"\n📈 Download Results Summary")
            print("=" * 50)
            print(f"Total symbols requested: {total}")
            print(f"Successfully processed: {successful}")
            print(f"Success rate: {successful/total*100:.1f}%")
            
            if successful > 0:
                print(f"\n✅ Successfully processed:")
                for symbol in sorted(results.keys()):
                    data = results[symbol]
                    print(f"  • {symbol}: {len(data)} days of data")
            
            failed = set(symbols) - set(results.keys())
            if failed:
                print(f"\n❌ Failed to process:")
                for symbol in sorted(failed):
                    print(f"  • {symbol}")
            
            # Show next steps
            print(f"\n🚀 Next Steps:")
            print(f"  • Run backtesting: python scripts/run_backtesting.py")
            print(f"  • Check data status: python scripts/data_collection.py --status")
            print(f"  • View results in: data/processed/model_inputs/")
        
        logger.info("Data collection script completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.debug("Full error traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())