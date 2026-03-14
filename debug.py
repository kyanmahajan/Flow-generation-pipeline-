import sys
import os

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Attempting to import moge...")
import moge.model.v2
print("SUCCESS! MoGe imported correctly.")