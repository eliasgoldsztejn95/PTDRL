#!/usr/bin/env python3
import rospy
from obstacle_detector.msg import Obstacles
from obstacle_detector.msg import CircleObstacle, SegmentObstacle
import numpy as np 

num_obstacles = 5
default_radius = 0.5
default_pos = 15

pub = rospy.Publisher('filtered_obstacles', Obstacles, queue_size=10)


def callback(msg):
    rate = rospy.Rate(10) # 10hz
    rate.sleep()
    filtered_obstacles = filter_obstacles(msg)
    pub.publish(filtered_obstacles)
    

def filter_obstacles(msg):
    # Select the closest obstacles. If it is a line segment it is translated into a circle with 0 velocity.

    filtered_obstacles = Obstacles()

    list_circles = msg.circles + lines_to_circles(msg.segments)

    sorted_list = sorted(list_circles, key=lambda obstacle: np.sqrt(obstacle.center.x**2 + obstacle.center.y**2))
    n_obstacles = sorted_list[0:num_obstacles - 1]
    
    for i in range(0, len(n_obstacles), 1):
        circle = CircleObstacle()

        circle.center.x = n_obstacles[i].center.x
        circle.center.y = n_obstacles[i].center.y

        circle.velocity.x = n_obstacles[i].velocity.x
        circle.velocity.y = n_obstacles[i].velocity.y

        circle.radius = n_obstacles[i].radius

        filtered_obstacles.circles.append(circle)

    for i in range(len(n_obstacles), num_obstacles - 1, 1):
        circle = CircleObstacle()

        circle.center.x = default_pos
        circle.center.y = default_pos

        circle.radius = default_radius
        filtered_obstacles.circles.append(circle)

    return filtered_obstacles


def lines_to_circles(list_lines):
    # Transform lines to circles. Center is closest x,y point on the line. Velocity is 0. default radius is 0.5.

    list_lines_to_circles = []
    for i in range(len(list_lines)):
        circle = CircleObstacle()
        x, y = closest_point_segment(list_lines[i])

        circle.center.x = x
        circle.center.y = y

        circle.velocity.x = 0
        circle.velocity.y = 0

        circle.radius = default_radius

        list_lines_to_circles.append(circle)  

    return list_lines_to_circles


def closest_point_line(line):
    # Given an infinite line, find the closest point to (0,0)

    x1 = line.first_point.x
    y1 = line.first_point.y

    x2 = line.last_point.x
    y2 = line.last_point.y

    a = y1 - y2
    b = x2 - x1

    c = x1*y2 - y1*x2

    x = (-a*c)/(a**2 + b**2)
    y = (-b*c)/(a**2 + b**2)

    return x, y

def closest_point_segment(line):
    # Given a segment, find the closest point to (0,0), in the segment.

    x1, y1 = closest_point_line(line)

    x2 = line.first_point.x
    y2 = line.first_point.y

    x3 = line.last_point.x
    y3 = line.last_point.y

    if not is_inside(x1,x2,x3):
        x1, y1 = 100, 100

    points = []
    points.append((x1, y1))
    points.append((x2, y2))
    points.append((x3, y3))

    distances = []
    distances.append(np.sqrt(x1**2 + y1**2))
    distances.append(np.sqrt(x2**2 + y2**2))
    distances.append(np.sqrt(x3**2 + y3**2))

    index_min = min(range(len(distances)), key=distances.__getitem__)

    return points[index_min]

def is_inside(x1,x2,x3):
    # Check if closest point in line is inside the segment
    inside = False
    if x2 > x3:
        if (x1 > x2) and (x1 < x3):
            inside = True
    if x3 < x2:
        if (x1 > x3) and (x1 < x2):
            inside = True
    return inside

def main():

    rospy.init_node('filtered_obstacles')
    #pub = rospy.Publisher('observation', Obstacles, queue_size=10)
    
    rospy.Subscriber("obstacles", Obstacles, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main()