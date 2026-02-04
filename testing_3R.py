from gui import ManipulatorInterface
import time

manipulator = ManipulatorInterface()

# Initialize to home
manipulator.init()
time.sleep(1)

# Move to some angles
manipulator.angle(30, 45, 60)
time.sleep(1)

# Move to a position
manipulator.pos(2, 1)
time.sleep(1)

# Back to home
manipulator.angle(0, 0, 0)
