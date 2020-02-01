import pybullet as p
import time
import pybullet_data
import os, inspect
import numpy as np
import copy
import math
import random

class tm700:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 1
    self.useSimulation = True
    self.useNullSpace = 21
    self.useOrientation = 0
    self.tmEndEffectorIndex = 5
    self.tmGripperIndex = 7
    # lower limits for null space
    # self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    self.ll = [-10, -10, -10, -10, -10, -10, -10]

    # upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    self.ul = [10, 10, 10, 10, 10, 10, 10]
    # joint ranges for null space
    # self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    self.jr = [10, 10, 10, 10, 10, 10, 10]
    # restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    # joint damping coefficents
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001, 0.00001, 0.00001
    ]
    self.reset()

  def reset(self):

    robot = p.loadURDF("../Gazebo_arm/urdf/tm700_robot_clean.urdf")
    self.tm700Uid = robot
    p.resetBasePositionAndOrientation(self.tm700Uid, [0.0, 0.0, 0.0], # position of robot, GREEN IS Y AXIS
                                      [0.000000, 0.000000, 1.000000, 0.000000]) # direction of robot
    self.jointPositions = [
        0.0, 0.0, -0, -0, -0.5, -1, -1.57, 0,
        #0.09196934635505513, -1.2212455855949105, -0.5971444720831858, -0.6572313840869254, -1.4991674243259474, 0.0,
         0.0, 0.0, -0, -0, -0, -0, -0, -0.0, -0.0, -0.0]

    self.numJoints = p.getNumJoints(self.tm700Uid)
    for jointIndex in range(1,self.numJoints):
        p.resetJointState(self.tm700Uid, jointIndex, self.jointPositions[jointIndex])
        p.setJointMotorControl2(self.tm700Uid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)

        # print('Link:', p.getLinkState(self.tm700Uid, jointIndex))

        # print(p.getJointInfo(robot, jointIndex))


    #
    # self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.6400, #first 3: position, last 4: quaternions
    #                           0.0000, 0.001, 0.000000, 0.000000, 1.000000, 0.000000)
    self.endEffectorPos = [0.0, 0.0, 0.0]
    self.endEffectorAngle = 0


# BLOCK
    xpos = 0.55   # * random.random()
    ypos =  0.2  # * random.random()
    ang = 3.14  #* random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF(os.path.join(self.urdfRootPath, "block.urdf"), xpos, ypos, 0.05,
                               orn[0], orn[1], orn[2], orn[3])
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    print('BLOCK INFO:', blockPos, blockOrn)
    print('block:', self.blockUid)

    self.motorNames = []
    self.motorIndices = []

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.tm700Uid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)
        # print('motorindeces', self.motorIndices)

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

    jointInfo = p.getJointInfo(self.tm700Uid, i)
    qIndex = jointInfo[3]

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.tm700Uid, self.tmGripperIndex)
    pos = state[0]
    orn = state[1] #Cartesian orientation of center of mass, in quaternion [x,y,z,w]

    euler = p.getEulerFromQuaternion(orn)

    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation

  def applyAction(self, motorCommands):

    #print ("self.numJoints")
    #print (self.numJoints)
    if (self.useInverseKinematics):

      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      da = motorCommands[3]
      fingerAngle = motorCommands[4]
      state = p.getLinkState(self.tm700Uid, self.tmEndEffectorIndex) # returns 1. center of mass cartesian coordinates, 2. rotation around center of mass in quaternion
      actualEndEffectorPos = state[0]
      #print("pos[2] (getLinkState(tmEndEffectorIndex)")
      #print(actualEndEffectorPos[2])

      self.endEffectorPos[0] =  dx
      self.endEffectorPos[1] = dy
      self.endEffectorPos[2] = dz
  #
      self.endEffectorAngle = self.endEffectorAngle + da
      pos = [dx, dy, dz]
      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])

      if (self.useOrientation == 1): # FALSE
        jointPoses = p.calculateInverseKinematics(self.tm700Uid,
                                                self.tmEndEffectorIndex,
                                                pos,
                                                orn,
                                                jointDamping=self.jd)
      else:
        jointPoses = p.calculateInverseKinematics(self.tm700Uid, self.tmEndEffectorIndex, pos,residualThreshold= 0.01)
        print('POSES OF JOINTS', jointPoses)

      if (self.useSimulation):
        for i in range(self.tmEndEffectorIndex+1):

          p.setJointMotorControl2(bodyUniqueId=self.tm700Uid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)

  def print_joint_state(self):
    print(p.getLinkState(self.tm700Uid, self.tmEndEffectorIndex))
    # print(p.getJointInfo(self.tm700Uid, 7))

if __name__ == '__main__':


    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

    tm700test = tm700()
    tm700test.reset
    p.setGravity(0,0,0)
    #tm700test.applyAction([0.67, 0.2, 0.05,0,0])
    tm700test.applyAction([0.55, 0.2, 0.05,0,0])
    for i in range (10000):
        p.stepSimulation()
        tm700test.print_joint_state()
        time.sleep(1./240.0)
    p.disconnect()
