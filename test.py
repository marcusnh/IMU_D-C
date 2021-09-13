import skinematics as skin

from skinematics.sensors.xsens import XSens
from skinematics.sensors.manual import MyOwnSensor
import numpy as np

skin.imus.IMU_Base.get_data('Data/testfile.csv')

# Set the in-file, initial sensor orientation
in_file = r'Data/test_skins.txt'
initial_orientation = np.array([[1,0,0],
                                [0,0,-1],
                                [0,1,0]])

# Only read in the data
data = XSens(in_file, q_type=None)

# Read in and evaluate the data
sensor = XSens(in_file=in_file, R_init=initial_orientation)

# By default, the orientation quaternion gets automatically calculated,
#    using the option "analytical"
q_analytical = sensor.quat

# Automatic re-calculation of orientation if "q_type" is changed
sensor.set_qtype('madgwick')
q_Madgwick = sensor.quat

sensor.set_qtype('kalman')
q_Kalman = sensor.quat

# Demonstrate how to fill up a sensor manually
in_data = {'rate':sensor.rate,
        'acc': sensor.acc,
        'omega':sensor.omega,
        'mag':sensor.mag}
my_sensor = MyOwnSensor(in_file='My own 123 sensor.', in_data=in_data)