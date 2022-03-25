#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

from __future__ import print_function

import argparse
from asyncio import DatagramTransport
from cProfile import label
from ctypes import sizeof
import logging
import random
import time
import cv2 as cv 
import os
import open3d as o3d
import sys

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MINI_WINDOW_WIDTH = 400
MINI_WINDOW_HEIGHT = 300
MAX_DISTANCE = 1000

#function to empty a directory
def emptydir(top):
    if(top == '/' or top == "\\"): return
    else:
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))  

def get_object_roi(image, step, label):
    roi = np.where(image == label) # 4 = pedestrian, 7 = road, 10 = cars
    #print(roi)
    try:
        box_min_width = np.min(roi[1])
        box_max_width = np.max(roi[1])
        box_min_height = np.min(roi[0])
        box_max_height = np.max(roi[0])

        print("Size of ROI is: {0} x {1}".format(box_max_height-box_min_height, box_max_width-box_min_width))
        return box_max_height, box_min_height, box_max_width, box_min_width

    except Exception as e:
        print("Error getting ROI in get_object_roi function: {}".format(e), file = sys.stderr)   

def measure_distance(depth_image, point_x, point_y):
    print(point_x, point_y)
    distance = depth_image[point_y, point_x]
    return distance

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=15,
        NumberOfPedestrians=30,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(2.2, 0.0, 0.8)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    camera1.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera1.set_position(2.2, 0.0, 0.8)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('CameraSemantic', PostProcessing='SemanticSegmentation')
    camera2.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera2.set_position(2.2, 0.0, 0.8)
    camera2.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera2)
    if args.lidar:
        lidar = sensor.Lidar('Lidar32')
        lidar.set_position(0, 0, 2.5)
        lidar.set_rotation(0, 0, 0)
        lidar.set(
            Channels=32, 
            Range=50,
            PointsPerSecond=100000,
            RotationFrequency=10,
            UpperFovLimit=0,
            LowerFovLimit=-1)
        settings.add_sensor(lidar)
    return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._display_map = args.map
        self._city_name = None
        self._map = None
        self._map_shape = None
        self._map_view = None
        self._position = None
        self._agent_positions = None

    def execute(self):
        """Launch the PyGame."""
        emptydir('D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image') # clear output folder each run
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        self._on_new_episode()

        if self._city_name is not None:
            self._map = CarlaMap(self._city_name, 0.1643, 50.0)
            self._map_shape = self._map.map_image.shape
            self._map_view = self._map.get_map(WINDOW_HEIGHT)

            extra_width = int((WINDOW_HEIGHT/float(self._map_shape[0]))*self._map_shape[1])
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + extra_width, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')

    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        if self._display_map:
            self._city_name = scene.map_name
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):
        self._timer.tick()
        measurements, sensor_data = self.client.read_data()

        self._main_image = sensor_data.get('CameraRGB', None)
        self._mini_view_image1 = sensor_data.get('CameraDepth', None)
        self._mini_view_image2 = sensor_data.get('CameraSemantic', None)
        self._lidar_measurement = sensor_data.get('Lidar32', None)
        
        #Functions relating to camera data and converting it to opencv and numpy data for processing
        if self._main_image is not None and self._mini_view_image1 is not None and self._mini_view_image2 is not None:
            label = 10 #object to outline. 4 = pedestrian, 7 = road, 10 = car
            
            image = image_converter.to_bgra_array(self._main_image)
            seg_image = image_converter.labels_to_array(self._mini_view_image2)
            #seg_image = cv.resize(seg_image, (WINDOW_WIDTH,WINDOW_HEIGHT), interpolation=cv.INTER_NEAREST)
            depth_image = MAX_DISTANCE*(image_converter.depth_to_array(self._mini_view_image1))
           
            #cv.imwrite('D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image\image{}.jpg'.format(self._timer.step), image)
           
            #np.savetxt('D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image\depth_image{}.csv'.format(self._timer.step), depth_image, delimiter = ',')
            #cv.imwrite('D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image\depth_image{}.jpg'.format(self._timer.step), depth_image)
           
            #np.savetxt('D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image\segmentation_image{}.csv'.format(self._timer.step), seg_image, delimiter = ',')
            #cv.imwrite('D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image\segmentation_image{}.jpg'.format(self._timer.step), seg_image)
            
            #cv.imshow("Image", image)
            #cv.imshow("Depth Image", depth_image)
            try: #get bounding box around object of interest
                box_max_height, box_min_height, box_max_width, box_min_width = get_object_roi(seg_image, self._timer.step, label)
                cv.rectangle(image, (box_min_width, box_min_height),(box_max_width,box_max_height), (0,0,0), 2)
                #cv.imshow("Image", image)
            except TypeError as e:
                print("Error calling get_object_roi function: {}".format(e), file = sys.stderr)
            
            try: #get ceterpoint of ROI, draw crosshair, get distance from centerpoint to camera
                average_width = int((box_max_width + box_min_width) / 2)
                average_height = int((box_max_height + box_min_height) / 2)

                cv.line(image, (average_width - 10,average_height) , (average_width + 10,average_height), (0,255,0), 1)
                cv.line(image, (average_width,average_height - 10) , (average_width,average_height + 10), (0,255,0), 1)

                distance = measure_distance(depth_image, average_width, average_height) 
                print("Distance to object is: {}m".format(distance))
                average_height = 0
                average_width = 0
            except Exception as e:
                print("Error calling measure_distance function: {}".format(e), file = sys.stderr)
            except: print ("Object not in frame!") 

        #Functions relating to the LiDAR, it's features and extracting the data for processing
        if self._lidar_measurement is not None:
            # lidar_data = self._lidar_measurement.point_cloud
            # o3d.io.write_point_cloud("D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image\PointCloud{}.pcd".format(self._timer.step), lidar_data)
            # #lidar_channels = self._lidar_measurement.channels
            # lidar_angle = self._lidar_measurement.horizontal_angle           
            # #print("Channels: {}\t". format(lidar_channels))
            # print("Angle at step {}: {}".format(self._timer.step, lidar_angle))

            # try:
            #     lidar_data_cloud = o3d.io.read_point_cloud(lidar_data)
            #     #o3d.visualization.draw_geometries([lidar_data_cloud])
            #     lidar_array = np.asarray(lidar_data_cloud)
            #     print("Lidar data: {}".format(lidar_array))
            # except TypeError:
            #     print("PointCloud is not iterable!")
            # except:
            #     ("Lidar data not sanitized!")
            lidar_data = np.array(self._lidar_measurement.data[:, :2])
            np.savetxt('D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image\lidar_raw{}.csv'.format(self._timer.step), lidar_data, delimiter = ',')
            lidar_data *= 2.0
            lidar_data += 100.0
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            #print(np.shape(lidar_data))
            np.savetxt('D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image\lidar_clean{}.csv'.format(self._timer.step), lidar_data, delimiter = ',')

            #print(np.size(lidar_data_clean))
            #lidar_array = np.asarray(lidar_data_clean)
            #lidar_size = np.size(lidar_array)
            #print('Lidar output size:{}'.format(lidar_size))
            #np.savetxt('D:\ADP\CarlaSimulator-20220211T045029Z-001\CarlaSimulator\PythonClient\estadp\out_image\lidar{}.csv'.format(self._timer.step), lidar_array, delimiter = ',')
        
        # Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)
            else:
                self._print_player_measurements(measurements.player_measurements)
            
            # Plot position on the map as well.

            self._timer.lap()

        control = self._get_keyboard_control(pygame.key.get_pressed())
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        control.reverse = self._is_on_reverse
        return control

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        # if self._mini_view_image1 is not None:
        #     array = image_converter.depth_to_logarithmic_grayscale(self._mini_view_image1)
        #     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        #     self._display.blit(surface, (gap_x, mini_image_y))

        # if self._mini_view_image2 is not None:
        #     array = image_converter.labels_to_cityscapes_palette(
        #         self._mini_view_image2)
        #     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        #     self._display.blit(
        #         surface, (2 * gap_x + MINI_WINDOW_WIDTH, mini_image_y))

        if self._lidar_measurement is not None:
            lidar_data = np.array(self._lidar_measurement.data[:, :2])
            lidar_data *= 2.0
            lidar_data += 100.0
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            #draw lidar
            lidar_img_size = (200, 200, 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            surface = pygame.surfarray.make_surface(lidar_img)
            self._display.blit(surface, (10, 10))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])

                    w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
                    h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster')
    argparser.add_argument(
        '-m', '--map',
        action='store_true',
        help='plot the map of the current city')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
