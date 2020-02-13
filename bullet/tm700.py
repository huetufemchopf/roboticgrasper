#code used from the following sources: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py,
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kuka.py


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
    self.useSimulation = 1
    self.useNullSpace = 1
    self.useOrientation = 0
    self.tmEndEffectorIndex = 6
    self.tmGripperIndex = 6
    self.tmFingerIndexL = 8
    self.tmFingerIndexR = 9 # not clear whether right and left is correct
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
    self.jd = None
    #     [
    #     0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
    #     0.00001, 0.00001, 0.00001, 0.00001
    # ]
    self.reset()

  def reset(self):

    robot = p.loadURDF("./Gazebo_arm/urdf/tm700_robot_clean.urdf") #add two dots to start it from pycharm. i have no idea why. TODO:
    self.tm700Uid = robot
    p.resetBasePositionAndOrientation(self.tm700Uid, [0.0, 0.0, -0.0], # position of robot, GREEN IS Y AXIS
                                      [0.000000, 0.000000, 1.000000, 0.000000]) # direction of robot
    self.jointPositions = [
        0.0, -0, -1.5, -0.0, -1.6, -0, -0, 1.5, -0.02,0.02] # position 6 is actually gripper joint

    self.numJoints = p.getNumJoints(self.tm700Uid)
    for jointIndex in range(self.numJoints):
        p.resetJointState(self.tm700Uid, jointIndex, self.jointPositions[jointIndex])
        p.setJointMotorControl2(self.tm700Uid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)

    self.endEffectorPos = [0.4317596244807792, 0.1470447615125933, 0.2876258566462587]
    self.endEffectorAngle = 0.02

    self.motorNames = []
    self.motorIndices = []

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.tm700Uid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:

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


      self.endEffectorPos[0] = self.endEffectorPos[0] + dx

      self.endEffectorPos[1] =  self.endEffectorPos[1] +  dy

      self.endEffectorPos[2] = self.endEffectorPos[2] +  dz
  #
      self.endEffectorAngle = self.endEffectorAngle + da
      pos = self.endEffectorPos
      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
      if (self.useNullSpace == 1):
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.tm700Uid, self.tmEndEffectorIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp)
        else:
          jointPoses = p.calculateInverseKinematics(self.tm700Uid,
                                                    self.tmEndEffectorIndex,
                                                    pos)
                                                    # lowerLimits=self.ll,
                                                    # upperLimits=self.ul,
                                                    # jointRanges=self.jr,
                                                    # restPoses=self.rp)
      else:
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.tm700Uid,
                                                    self.tmEndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd)
        else:
          jointPoses = p.calculateInverseKinematics(self.tm700Uid, self.tmEndEffectorIndex, pos)

      if (self.useSimulation):
        for i in range(self.tmEndEffectorIndex):

          p.setJointMotorControl2(bodyUniqueId=self.tm700Uid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.tm700Uid, i, jointPoses[i])


      p.setJointMotorControl2(self.tm700Uid,
                          8,
                          p.POSITION_CONTROL,
                          targetPosition=-fingerAngle/4.,
                          force=self.fingerTipForce)

      p.setJointMotorControl2(self.tm700Uid,
                          9,
                          p.POSITION_CONTROL,
                          targetPosition=fingerAngle/4.,
                          force=self.fingerTipForce)


    else:
      for action in range(len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.tm700Uid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)

  def print_joint_state(self):
    pass

  def grasping(self):

    p.setJointMotorControl2(self.tm700Uid,
                          8,
                          p.POSITION_CONTROL,
                          targetPosition=0,
                          force=self.fingerTipForce)
    p.setJointMotorControl2(self.tm700Uid,
                          9,
                          p.POSITION_CONTROL,
                          targetPosition=0,
                          force=self.fingerTipForce)



if __name__ == '__main__':


    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    tm700test = tm700()
    # tm700test.reset
    p.setGravity(0,0,-10)
    # tm700test.applyAction([0.67, 0.2, 0.05,1,0.1])
    for i in range (1000):
        p.stepSimulation()
        # tm700test.print_joint_state()
        time.sleep(1./240.0)
    # tm700test.grasping()
    # tm700test.applyAction([0.67, 0.2, 0.15,0.2,0.20])
    # for i in range (10000):
    #     p.stepSimulation()
    #     # tm700test.print_joint_state()
        time.sleep(1./240.0)


    p.disconnect()
