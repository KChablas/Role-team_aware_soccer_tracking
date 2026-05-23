import sys
sys.path.insert(0, '.')

print('Testing nms...', end=' ')
from src.nms import my_non_max_suppression
print('OK')

print('Testing camera_estimator...', end=' ')
from src.camera_estimator import CameraEstimator
print('OK')

print('Testing mot_writer...', end=' ')
from src.mot_writer import MOTWriter
print('OK')

print('Testing team_assigner...', end=' ')
from src.team_assigner import TeamAssignerV2
print('OK')

print('Testing detector (requires ultralytics)...', end=' ')
from src.detector import YOLODetector
print('OK')

print('Testing tracker (requires boxmot)...', end=' ')
from src.tracker import StrongSortTracker
print('OK')

print('Testing sequence_runner...', end=' ')
from src.sequence_runner import SequenceRunner
print('OK')

print('Testing main...', end=' ')
import src.main
print('OK')

print()
print('All imports successful -- no circular dependencies.')
