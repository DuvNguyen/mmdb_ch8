import sys
import os

# Set path to current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import run

def main():
    print("=" * 60)
    print("  AUDIO DATABASE SYSTEM - Chuong 8 GUI")
    print("=" * 60)
    
    # Launch GUI
    run()

if __name__ == "__main__":
    main()
