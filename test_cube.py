import cozmo
import time


def cozmo_program(robot: cozmo.robot.Robot):
    robot.enable_device_imu(True, True, True)
    while True:
        yaw = robot.pose.rotation.angle_z.radians
        position = [robot.pose.position.x / 1000, robot.pose.position.y / 1000,
                    robot.pose.position.z / 1000, 0, 0, yaw]
        print("Cozmo yaw", yaw)
        print("Cozmo position", position)
        cube = robot.world.wait_until_observe_num_objects(num=1, object_type=cozmo.objects.LightCube, timeout=1)
        if cube:
            cube_yaw = cube[0].pose.rotation.angle_z.radians
            cube_position = [cube[0].pose.position.x / 1000, cube[0].pose.position.y / 1000,
                             cube[0].pose.position.z / 1000, 0, 0, cube_yaw]
            print("Cube yaw", cube_yaw)
            print("Cube position", cube_position)
        time.sleep(2)


cozmo.run_program(cozmo_program, use_viewer=True)
